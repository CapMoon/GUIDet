import torch
import cv2
import numpy as np
import os
import json
from tqdm import tqdm

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# 获取标注文件图像id与图像名字的字典
def get_name2id_map():

    # 获取标注文件的images标注信息
    anno_json = r'D:/yolov5_new/coco/annotations/annotations.json'
    with open(anno_json, 'r') as fr:
        anno_dict = json.load(fr)
    image_dict = anno_dict['images']

    # 构建图像名称与索引的字典对
    name2id_dict = {}
    for image in image_dict:
        # file_name = image['file_name'].split('.')[0]    # maksssksksss98.png -> maksssksksss98
        file_name = image['file_name']
        id = image['id']
        name2id_dict[file_name] = id

    return name2id_dict


# 功能：单图像推理
def val(image_dir, img_size=640, stride=32, augment=False, visualize=False):

    device = 'cpu'
    weights = r'D:/yolov5_new/weights/yolov5_regular.pt'  # 权重路径
    anno_json = r'D:/yolov5_new/coco/annotations/annotations.json'     # 已处理的标注信息json文件
    pred_json = 'preditions.json'                   # 带保存的预测信息json文件

    # 导入模型
    model = attempt_load(weights, device=device)
    img_size = check_img_size(img_size, s=stride)
    # names = model.names

    jdict = []
    name2id_dict = get_name2id_map()
    image_list = os.listdir(image_dir)

    # 依次预测每张图像，将预测信息全部保存在json文件中
    for image_name in tqdm(image_list, desc='val image'):

        # Padded resize
        image_path = image_dir + os.sep + image_name
        img0 = cv2.imread(image_path)
        img = letterbox(img0, img_size, stride=stride, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0   # 0 - 255 to 0.0 - 1.0
        img = img[None]     # [h w c] -> [1 h w c]

        # inference
        pred = model(img, augment=augment, visualize=visualize)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

        # plot label
        det = pred[0]
        # annotator = Annotator(img0.copy(), line_width=3, example=str(names))

        if len(det):
            # (xyxy, conf, cls)
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            # bbox格式需要转换: xyxy -> (x_min, y_min, width, height)
            bbox = xyxy2xywh(det[:, :4])
            bbox[:, :2] -= bbox[:, 2:] / 2  # xy center to top-left corner
            score = det[:, 4]
            category_id = det[:, -1]

            for box, src, cls in zip(bbox, score, category_id):
                jdict.append(
                    {'image_id': name2id_dict[image_name],
                     'category_id': int(cls),
                     'bbox': box.tolist(),
                     'score': float(src)}
                )

    # 保存预测好的json文件
    with open(pred_json, 'w') as fw:
        json.dump(jdict, fw, indent=4, ensure_ascii=False)

    # 使用coco api评价指标
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')

    eval.evaluate()
    eval.accumulate()
    eval.summarize()


if __name__ == '__main__':

    image_dir = r'D:/yolov5_new/datasets/images/val'
    val(image_dir=image_dir)
