#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from lxml.etree import Element, SubElement, tostring, ElementTree

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# classes = ["long_tube", "short_tube", "centrifuge_tube",
#            "plastic_tube", "trace_cup", "red_cap", "orange_cap", "yellow_cap", "green_cap", "purple_cap", "barcode", "liquid"]  # 类别


classes = ["blood_tube", "5ML_centrifuge_tube", "10ML_centrifuge_tube", "5ML_sorting_tube_rack", "10ML_sorting_tube_rack", "centrifuge_open", "centrifuge_close",
           "refrigerator_open", "refrigerator_close", "operating_desktop", "tobe_sorted_tube_rack", "dispensing_tube_rack", "sorting_tube_rack_base",
           "tube_rack_storage_cabinet"]  # 类别

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0    # (x_min + x_max) / 2.0
    y = (box[2] + box[3]) / 2.0    # (y_min + y_max) / 2.0
    w = box[1] - box[0]   # x_max - x_min
    h = box[3] - box[2]   # y_max - y_min
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('E:/ABB/AI/yolov9/data/Annotations/%s.xml' % (image_id), encoding='UTF-8')

    out_file = open('E:/ABB/AI/yolov9/data/labels1/%s.txt' % (image_id), 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # print(cls)
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

xml_path = os.path.join(CURRENT_DIR, 'E:/ABB/AI/yolov9/data/Annotations/')

# xml list
img_xmls = os.listdir(xml_path)
for img_xml in img_xmls:
    label_name = img_xml.split('.')[0]
    print(label_name)
    convert_annotation(label_name)
