'''
修正脚本：对预测的json文件还有标注的json文件的id信息根据标注文件的image来命名
'''

import json
import os
from collections import OrderedDict

# 获取标注文件图像id与图像名字的字典
def get_name2id_map(image_dict):

    name2id_dict = OrderedDict()
    for image in image_dict:
        file_name = image['file_name'].split('.')[0]    # maksssksksss98.png -> maksssksksss98
        id = image['id']
        name2id_dict[file_name] = id

    return name2id_dict


if __name__ == '__main__':

    anno_json = r'D:/yolov5_new/coco/annotations/annotations.json'
    pred_json = r'D:/yolov5_new/coco/predictions/yolov5_regular_predictions.json'

    with open(pred_json, 'r') as fr:
        pred_dict = json.load(fr)
    with open(anno_json, 'r') as fr:
        anno_dict = json.load(fr)

    name2id_dict = get_name2id_map(anno_dict['images'])

    # 对标注文件annotations的image_id进行更改
    for annotations in anno_dict['annotations']:
        image_id = annotations['image_id']
        annotations['image_id'] = int(name2id_dict[image_id])

    # 对预测文件的image_id同样进行更改
    for predictions in pred_dict:
        image_id = predictions['image_id']
        predictions['image_id'] = int(name2id_dict[image_id])

    # 分别保存更改后的标注文件和预测文件
    with open('anno_json.json', 'w') as fw:
        json.dump(anno_dict, fw, indent=4, ensure_ascii=False)
    with open('pred_json.json', 'w') as fw:
        json.dump(pred_dict, fw, indent=4, ensure_ascii=False)



