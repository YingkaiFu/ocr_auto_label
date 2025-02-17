import json
import numpy as np
import mmcv
import os
import cv2
import math
from PyQt5.QtCore import QPointF, QPoint
from shapely.geometry import Polygon
from typing import Tuple, List, Dict
from mmcv import imnormalize
from dicts import COMMON_DICT



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.ndarray)):
            return super(NpEncoder, self).default(obj) if isinstance(obj, np.ndarray) else obj.item()
        
        # 添加对QPoint的支持
        if isinstance(obj, QPoint or QPointF):
            return [obj.x(),  obj.y()]
        
        # 如果obj是一个自定义类型，可以在这里添加更多的类型处理逻辑
        
        return super(NpEncoder, self).default(obj)


def save_json_files(json_path, image, data):
    image_data = mmcv.imread(str(image))
    image_height, image_width, _ = image_data.shape
    basic_info = {
    	"version": "by rolabeimg convert",
        "flags": {},
        "imagePath": image.name,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "shapes": []
    }
    det_labeled = len(data)
    rec_labeled = 0
    for rec_info in data:
        class_name = rec_info["category"]
        rect_points = rec_info["points"]
        rec_label = rec_info.get("rec_label", "")
        if rec_label !="" and rec_label is not None:
            rec_labeled = rec_labeled + 1
        shape_info = {
            "label": class_name,
            "points": rect_points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": rec_label
        }
        basic_info["shapes"].append(shape_info)
    with open(json_path, 'w') as f:
        json.dump(basic_info, f, indent=4,cls=NpEncoder)
    
    return det_labeled, rec_labeled

def load_json_files(json_path,color_map):
    rect_infos = []
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        for shape in json_data["shapes"]:
            class_name = shape["label"]
            rect_points = shape["points"]
            rect_qpoints = [QPointF(x, y) for x, y in rect_points]
            rect_label = shape.get("flags", "")
            if rect_label == {} or rect_label is None:
                rect_label = ""
            rect_infos.append({
                "category": class_name,
                "points": rect_qpoints,
                "color": color_map[class_name],
                "rec_label": rect_label
            })
        return rect_infos

def checkout_det_rec_label(json_path):
    has_rec_label = True
    has_det_label = True
    if not os.path.exists(json_path):
        return False, False
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        if json_data["shapes"]:
            for shape in json_data["shapes"]:
                if not shape.get("points", ""):
                    has_det_label = False
                if not shape.get("flags", ""):
                    has_rec_label = False
        else:
            has_det_label = False
            has_rec_label = False

    return has_det_label, has_rec_label


def polygon_from_points(points):
    """Convert points to a Shapely Polygon."""
    return Polygon(points)

def calculate_iou(poly1, poly2):
    """Calculate the IoU of two polygons."""
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0
    return inter / union

def match_bboxes(gt_bboxes, pred_bboxes, iou_threshold=0.7):
    """Match ground truth bboxes with predicted bboxes based on IoU."""
    matches = []
    for gt_bbox in gt_bboxes:
        best_match = None
        best_iou = 0
        for pred_bbox in pred_bboxes:
            iou = calculate_iou(polygon_from_points(gt_bbox['points']), polygon_from_points(pred_bbox['points']))
            if iou > best_iou and iou >= iou_threshold:
                best_match = pred_bbox
                best_iou = iou
        if best_match is not None:
            matches.append((gt_bbox, best_match))
    return matches

def text_matches(gt_text, pred_text, ignore_case=True, ignore_punctuation=True):
    """Check if the texts match based on given criteria."""
    if ignore_case:
        gt_text = gt_text.lower()
        pred_text = pred_text.lower()
    if ignore_punctuation:
        gt_text = ''.join(e for e in gt_text if e.isalnum() or e.isspace())
        pred_text = ''.join(e for e in pred_text if e.isalnum() or e.isspace())
    return gt_text == pred_text

def calculate_accuracy(gt_data, pred_data):
    """Calculate overall accuracy of OCR detection."""
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    all_det_box_gt = 0
    all_det_box_pred = 0
    all_det_box_pred_correct = 0
    all_rec_box_pred_correct = 0
    for res_dict_gt, res_dict_pred in zip(gt_data, pred_data):
        matches = match_bboxes(res_dict_gt['det_results'], res_dict_pred['det_results'])
        
        # Count true positives and false negatives
        for gt_bbox, pred_bbox in matches:
            if text_matches(gt_bbox['rec_res'], pred_bbox['rec_res'][0], ignore_case=False,ignore_punctuation=False) or text_matches(gt_bbox['rec_res'], pred_bbox['rec_res'][1], ignore_case=False,ignore_punctuation=False):
                all_rec_box_pred_correct += 1
            else:
                print(res_dict_pred["image_path"], gt_bbox['rec_res'], pred_bbox['rec_res'][0],pred_bbox['rec_res'][1])
                fp += 1  # If the text doesn't match, it's considered as a false positive
        
        all_det_box_gt += len(res_dict_gt['det_results'])
        all_det_box_pred += len(res_dict_pred['det_results'])
        all_det_box_pred_correct += len(matches)

    
    return {
        "det_gt_num": all_det_box_gt,
        "det_pred_num": all_det_box_pred,
        "det_correct_num": all_det_box_pred_correct,
        "rec_correct_num": all_rec_box_pred_correct,
    }

def convert_image_direction(img: np.array, direction: str) -> np.array:
    if direction == "rotate_180":
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif direction == "rotate_180_mirror":
        img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.flip(img, 1)
    elif direction == "mirror":
        img = cv2.flip(img, 1)
    return img

def preprocess_image(img: np.array, input_shape: tuple, mean: np.array = np.array([127, 127, 127]), std: np.array = np.array([127, 127, 127]), to_rgb: bool = True, roi: List[List[float]] = None, return_img = False, direction='normal') -> np.array:
    """
    对输入图像进行预处理，包括透视变换、调整大小、归一化和格式转换。

    参数:
        img (np.array): 原始 BGR 格式图像。
        input_shape (tuple): 输入图像的形状。
        mean (np.array): 图像均值。
        std (np.array): 图像标准差。
        to_rgb (bool): 是否转换为 RGB 格式。
        roi (List[List[float]]): 图像的 ROI 坐标。
        return_img (bool): 是否返回预处理后的图像。
        direction (str): 图像处理方向。

    返回:
        np.array: 预处理后的图像张量。

    """
    if roi is not None:
        img = crop_perpective_transform(img, roi)

    img = convert_image_direction(img, direction)

    target_h, target_w = input_shape[2], input_shape[3]

    # 调整图像大小
    resized_img = resize_img(img, target_h, target_w)

    # 归一化图像
    img_normalized = imnormalize(resized_img, mean, std, to_rgb=to_rgb)

    # 转换为 (N, C, H, W) 格式
    img_transposed = img_normalized.transpose(2, 0, 1)[None, :]

    if return_img:
        return img_transposed, resized_img
    
    return img_transposed

def resize_img(img: np.array, target_h: int, target_w: int) -> np.array:
    """
    调整图像大小，同时保持纵横比并填充到目标尺寸。

    参数:
        img (np.array): 输入图像。
        target_h (int): 目标高度。
        target_w (int): 目标宽度。

    返回:
        np.array: 调整大小并填充后的图像。
    """
    h, w, c = img.shape
    ratio = target_h / h
    ow = math.ceil(ratio * w)
    ow = min(target_w, ow)
    ow = round(ow / 4) * 4  # 保证宽度为4的倍数

    # 调整图像大小
    img = cv2.resize(img, (ow, target_h), interpolation=cv2.INTER_AREA)

    # 填充图像
    padded_image = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
    padded_image[0:target_h, 0:ow] = img

    return padded_image

def find_corners(points):
    point_list = [points[i] for i in range(0, len(points))]
    point_sort_left = sorted(point_list, key=lambda k: [k[0], k[1]])
    lt, ld = sorted([point_sort_left[0], point_sort_left[1]], key=lambda k:[k[1], k[0]])
    rt, rd = sorted([point_sort_left[2], point_sort_left[3]], key=lambda k:[k[1], k[0]])
    return (lt, rt, rd, ld)
    
def crop_perpective_transform(image: np.array, points: List[List[float]]) -> np.array:
    """
    对图像进行透视变换，裁剪出感兴趣区域（ROI）。

    参数:
        image (np.array): 输入图像。
        points (List[List[float]]): 4个角点的坐标，依次为左上、右上、左下、右下。

    返回:
        np.array: 透视变换后的图像。
    """
    pts_src = np.array(points, dtype=np.float32)
    # 计算宽度和高度
    width = int(np.linalg.norm(pts_src[1] - pts_src[0]))
    height = int(np.linalg.norm(pts_src[3] - pts_src[0]))
    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    
    # 计算透视变换矩阵并应用变换
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))
    return transformed_image

def process_output(outputs: np.array, dictionary: List[str]) -> Dict[str, float]:
    """
    处理模型输出，生成预测文本并计算分数。

    参数:
        outputs (np.array): 模型输出，通常为二维概率数组。
        dictionary (List[str]): 字典，用于将索引映射到字符。

    返回:
        Dict[str, float]: 包含预测文本和平均得分的字典。
    """
    # 确保 outputs 是一个合适的 numpy 数组，并取第一个输出
    probs = np.asarray(outputs)[0]
    feat_len = probs.shape[0]  # 获取序列长度

    # 找到每行的最大值及其索引
    max_value = np.max(probs, axis=-1)
    max_idx = np.argmax(probs, axis=-1)

    index, score = [], []
    prev_idx = len(dictionary)  # 设置为字典长度，作为无效索引
    text = ''
    rs = []  # 存储所有的索引值

    for t in range(feat_len):
        tmp_value = int(max_idx[t])  # 获取当前索引
        
        # 如果当前值与上一个值不同，且不是无效索引，进行处理
        if tmp_value != prev_idx and tmp_value < len(dictionary):  
            index.append(tmp_value)
            score.append(float(max_value[t]))  # 保存最大值得分

        prev_idx = tmp_value  # 更新前一个索引
        rs.append(tmp_value)  # 保存所有索引值用于后续处理
    
    # 针对异常情况进行处理
    if not index:
        return dict(label_id=-1, label_name="无有效预测", score=0.0, segmentation=[[]])  # 没有有效预测时返回默认值
    
    # 使用字典构建输出文本
    for i in index:
        text += dictionary[i]

    # 计算平均分，如果 score 为空，则设置为 0
    avg_score = np.mean(score) if score else 0.0

    return dict(label_id=0, label_name=text, score=[round(float(avg_score), 4)], segmentation=[[]])


def parse_dict(dict_type: str) -> List[str]:
    """
    解析字典类型，并返回相应的字符列表。

    参数:
        dict_type (str): 字典类型，例如 "common"。

    返回:
        List[str]: 解析后的字符列表。
    """
    if dict_type == "common":
        dictionary = COMMON_DICT
    else:
        raise ValueError(f"未知的字典类型: {dict_type}")
    
    return dictionary.split("\n")