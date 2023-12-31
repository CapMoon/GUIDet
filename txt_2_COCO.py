import os
import json
import random
import time
from PIL import Image
import csv

coco_format_save_path = './coco'  # 要生成的标准coco格式标签所在文件夹
yolo_format_classes_path = 'annotations.csv'  # 类别文件,用csv文件表示，一行一个类
yolo_format_annotation_path = 'D:/yolov5_new/datasets/labels/val'  # yolo格式标签所在文件夹
img_pathDir = 'D:/yolov5_new/datasets/images/val'  # 图片所在文件夹
# with open(yolo_format_classes_path, 'r') as f:
#     reader = csv.reader(f)
#     for label in reader:
#         print(label)
# categories = []
# for i in label:
#     categories.append({'id': label.index(i) + 1, 'name': i, 'supercategory': ""})  # 存储类别

categories = []
class_names = ["Background Image", "Checked View", "Icon", "Input Field", "Image", "Text", "Text Button", "Sliding Menu", "Page Indicator", "Upper Task Bar", "Pop-Up Window", "Switch"]
for label in class_names:
    categories.append({'id': class_names.index(label), 'name': label, 'supercategory': ""})

write_json_context = dict()  # 写入.json文件的大字典
write_json_context['licenses'] = [{'name': "", 'id': 0, 'url': ""}]
write_json_context['info'] = {'contributor': "", 'date_created': "", 'description': "",
                              'url': "", 'version': "", 'year': ""}
write_json_context['categories'] = categories
write_json_context['images'] = []
write_json_context['annotations'] = []

# 接下来的代码主要添加'images'和'annotations'的key值
imageFileList = os.listdir(img_pathDir)
# 遍历该文件夹下的所有文件，并将所有文件名添加到列表中
img_id = 0  # 图片编号
anno_id = 0     # 标注标号
for i, imageFile in enumerate(imageFileList):
    if '_' not in imageFile:
        img_id += 1
        imagePath = os.path.join(img_pathDir, imageFile)  # 获取图片的绝对路径
        image = Image.open(imagePath)  # 读取图片
        W, H = image.size  # 获取图片的高度宽度
        img_context = {}  # 使用一个字典存储该图片信息
        # img_name=os.path.basename(imagePath)
        img_context['id'] = img_id  # 每张图像的唯一ID索引
        img_context['width'] = W
        img_context['height'] = H
        img_context['file_name'] = imageFile
        img_context['license'] = 0
        img_context['flickr_url'] = ""
        img_context['color_url'] = ""
        img_context['date_captured'] = ""

        write_json_context['images'].append(img_context)  # 将该图片信息添加到'image'列表中

        txtFile = imageFile.split('.')[0] + '.txt'  # 获取该图片获取的txt文件
        with open(os.path.join(yolo_format_annotation_path, txtFile), 'r') as fr:
            lines = fr.readlines()  # 读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息

        for j, line in enumerate(lines):
            anno_id += 1  # 标注的id从1开始
            bbox_dict = {}  # 将每一个bounding box信息存储在该字典中

            class_id, x, y, w, h = line.strip().split(' ')  # 获取每一个标注框的详细信息
            class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)  # 将字符串类型转为可计算的int和float类型

            # 坐标转换
            xmin = (x - w / 2) * W
            ymin = (y - h / 2) * H
            xmax = (x + w / 2) * W
            ymax = (y + h / 2) * H
            w = w * W
            h = h * H
            height, width = abs(ymax - ymin), abs(xmax - xmin)

            # bounding box的坐标信息
            bbox_dict['id'] = anno_id               # 每个标注信息的索引
            bbox_dict['image_id'] = img_id          # 当前图像的ID索引
            bbox_dict['category_id'] = class_id     # 类别信息
            bbox_dict['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
            bbox_dict['area'] = height * width
            bbox_dict['bbox'] = [xmin, ymin, w, h]  # 注意目标类别要加一
            bbox_dict['iscrowd'] = 0
            bbox_dict['attributes'] = ""

            write_json_context['annotations'].append(bbox_dict)  # 将每一个由字典存储的bounding box信息添加到'annotations'列表中

name = os.path.join(coco_format_save_path, "annotations" + '.json')
with open(name, 'w') as fw:  # 将字典信息写入.json文件中
    json.dump(write_json_context, fw, indent=4, ensure_ascii=False)
