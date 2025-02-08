import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将根目录添加到路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # 模型路径或 Triton URL
        source=ROOT / 'data/images',  # 文件/目录/URL/模式/屏幕/0（网络摄像头）
        data=ROOT / 'data/coco.yaml',  # 数据集.yaml 路径
        imgsz=(640, 640),  # 推理大小（高度，宽度）
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IOU 阈值
        max_det=1000,  # 每张图像的最大检测数量
        device='',  # CUDA 设备，例如 0 或 0,1,2,3 或 cpu
        view_img=False,  # 显示结果
        save_txt=False,  # 将结果保存到 *.txt
        save_conf=False,  # 在 --save-txt 标签中保存置信度
        save_crop=False,  # 保存裁剪的预测框
        nosave=False,  # 不保存图像/视频
        classes=None,  # 按类过滤： --class 0 或 --class 0 2 3
        agnostic_nms=False,  # 类别无关的 NMS
        augment=False,  # 增强推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 将结果保存到项目/名称
        name='exp',  # 将结果保存到项目/名称
        exist_ok=False,  # 允许存在的项目/名称，不增加
        line_thickness=3,  # 边界框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用 FP16 半精度推理
        dnn=False,  # 使用 OpenCV DNN 进行 ONNX 推理
        vid_stride=1,  # 视频帧率间隔
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # 保存推理图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # 下载

    # 目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增加运行编号
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小

    # 数据加载器
    bs = 1  # 批处理大小
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 预热
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 转 fp16/32
            im /= 255  # 0 - 255 转 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # 扩展为批处理维度

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 二级分类器（可选）
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 处理预测结果
        for i, det in enumerate(pred):  # 每张图像
            seen += 1
            if webcam:  # 批处理大小 >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # 转换为 Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益 whwh
            imc = im0.copy() if save_crop else im0  # 用于 save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 将框从 img_size 缩放到 im0 大小
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 每个类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # 写入文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化 xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # 将边框添加到图像
                        c = int(cls)  # 整数类
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 流结果
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小（Linux）
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 毫秒

            # 保存结果（带检测的图像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' 或 'stream'
                    if vid_path[i] != save_path:  # 新视频
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 释放上一个视频写入器
                        if vid_cap:  # 视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # 强制结果视频使用 *.mp4 后缀
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 打印时间（仅推理）
        LOGGER.info(f"{s}{'' if len(det) else '(无检测), '}{dt[1].dt * 1E3:.1f}ms")

    # 打印结果
    t = tuple(x.t / seen * 1E3 for x in dt)  # 每张图像的速度
    LOGGER.info(f'速度: %.1fms 预处理, %.1fms 推理, %.1fms NMS 每张图像，形状 {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} 个标签已保存到 {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"结果已保存到 {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 更新模型（以修复 SourceChangeWarning）


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp19/weights/best.pt', help='模型路径或 Triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/dataset-test', help='文件/目录/URL/模式/屏幕/0（网络摄像头）')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco-dataset.yaml', help='（可选）数据集.yaml 路径')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='推理大小 h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--max-det', type=int, default=1000, help='每张图像的最大检测数量')
    parser.add_argument('--device', default='', help='CUDA 设备，例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='将结果保存到 *.txt')
    parser.add_argument('--save-conf', action='store_true', help='在 --save-txt 标签中保存置信度')
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪的预测框')
    parser.add_argument('--nosave', action='store_true', help='不保存图像/视频')
    parser.add_argument('--classes', nargs='+', type=int, help='按类过滤： --classes 0 或 --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的 NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--visualize', action='store_true', help='可视化特征')
    parser.add_argument('--update', action='store_true', help='更新所有模型')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='将结果保存到项目/名称')
    parser.add_argument('--name', default='exp', help='将结果保存到项目/名称')
    parser.add_argument('--exist-ok', action='store_true', help='允许存在的项目/名称，不增加')
    parser.add_argument('--line-thickness', default=3, type=int, help='边界框厚度（像素）')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='隐藏置信度')
    parser.add_argument('--half', action='store_true', help='使用 FP16 半精度推理')
    parser.add_argument('--dnn', action='store_true', help='使用 OpenCV DNN 进行 ONNX 推理')
    parser.add_argument('--vid-stride', type=int, default=1, help='视频帧率间隔')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 扩展
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
