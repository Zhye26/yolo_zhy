#!/usr/bin/env python3
"""
数据清洗脚本
检测数据集中的漏标、错标问题
"""
import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict


def load_yolo_labels(label_path):
    """加载YOLO格式标注"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    labels.append({'class': cls, 'x': x, 'y': y, 'w': w, 'h': h})
    return labels


def xywh_to_xyxy(x, y, w, h, img_w, img_h):
    """YOLO格式转像素坐标"""
    x1 = (x - w/2) * img_w
    y1 = (y - h/2) * img_h
    x2 = (x + w/2) * img_w
    y2 = (y + h/2) * img_h
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    """计算IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def analyze_image(model, img_path, label_path, conf_thresh=0.3, iou_thresh=0.5):
    """
    分析单张图片的标注质量
    Returns:
        dict: 分析结果
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_h, img_w = img.shape[:2]

    # 加载标注
    gt_labels = load_yolo_labels(label_path)
    gt_boxes = []
    for lbl in gt_labels:
        box = xywh_to_xyxy(lbl['x'], lbl['y'], lbl['w'], lbl['h'], img_w, img_h)
        gt_boxes.append({'box': box, 'class': lbl['class'], 'matched': False})

    # 模型预测
    results = model(img, conf=conf_thresh, verbose=False)[0]
    pred_boxes = []
    if results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            pred_boxes.append({'box': xyxy, 'class': cls, 'conf': conf, 'matched': False})

    # 匹配预测和标注
    for pred in pred_boxes:
        best_iou = 0
        best_gt = None
        for gt in gt_boxes:
            if gt['matched']:
                continue
            if pred['class'] == gt['class']:
                iou = compute_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

        if best_iou >= iou_thresh and best_gt:
            pred['matched'] = True
            best_gt['matched'] = True

    # 统计问题
    issues = {
        'missing_labels': [],  # 漏标：模型检测到但没标注
        'extra_labels': [],    # 多标：有标注但模型没检测到
        'low_conf_matches': [] # 低置信度匹配
    }

    for pred in pred_boxes:
        if not pred['matched'] and pred['conf'] >= 0.5:
            issues['missing_labels'].append(pred)

    for gt in gt_boxes:
        if not gt['matched']:
            issues['extra_labels'].append(gt)

    return {
        'image': img_path,
        'gt_count': len(gt_boxes),
        'pred_count': len(pred_boxes),
        'issues': issues,
        'has_issues': len(issues['missing_labels']) > 0 or len(issues['extra_labels']) > 0
    }


def clean_dataset(model_path, data_dir, output_dir, conf_thresh=0.3):
    """
    清洗整个数据集
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 支持两种目录结构
    if os.path.exists(os.path.join(data_dir, 'images')):
        images_dir = os.path.join(data_dir, 'images')
        labels_dir = os.path.join(data_dir, 'labels')
    else:
        images_dir = data_dir
        labels_dir = data_dir.replace('/images', '/labels')

    if not os.path.exists(images_dir):
        print(f"错误: 图片目录不存在: {images_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 收集所有图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(images_dir).glob(ext))

    print(f"共 {len(image_files)} 张图片")

    # 分析每张图片
    results = []
    issues_count = defaultdict(int)
    class_names = ['ebike', 'driver', 'passenger', 'helmet']

    for i, img_path in enumerate(image_files):
        label_path = os.path.join(labels_dir, img_path.stem + '.txt')
        result = analyze_image(model, str(img_path), label_path, conf_thresh)

        if result:
            results.append(result)
            if result['has_issues']:
                issues_count['total'] += 1
                issues_count['missing'] += len(result['issues']['missing_labels'])
                issues_count['extra'] += len(result['issues']['extra_labels'])

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(image_files)}")

    # 筛选问题图片
    problem_images = [r for r in results if r['has_issues']]

    # 按问题严重程度排序
    problem_images.sort(key=lambda x: len(x['issues']['missing_labels']) + len(x['issues']['extra_labels']), reverse=True)

    # 生成报告
    report = {
        'summary': {
            'total_images': len(results),
            'problem_images': len(problem_images),
            'missing_labels': issues_count['missing'],
            'extra_labels': issues_count['extra']
        },
        'problem_images': problem_images[:100]  # 只保存前100个
    }

    report_path = os.path.join(output_dir, 'cleaning_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 生成问题图片列表
    problem_list_path = os.path.join(output_dir, 'problem_images.txt')
    with open(problem_list_path, 'w') as f:
        for img in problem_images:
            missing = len(img['issues']['missing_labels'])
            extra = len(img['issues']['extra_labels'])
            f.write(f"{img['image']}\t漏标:{missing}\t多标:{extra}\n")

    # 打印统计
    print("\n" + "=" * 60)
    print("数据清洗报告")
    print("=" * 60)
    print(f"总图片数: {len(results)}")
    print(f"问题图片: {len(problem_images)} ({len(problem_images)/len(results)*100:.1f}%)")
    print(f"  - 漏标数量: {issues_count['missing']}")
    print(f"  - 多标数量: {issues_count['extra']}")
    print(f"\n报告保存: {report_path}")
    print(f"问题列表: {problem_list_path}")

    return report


def visualize_issues(model_path, img_path, label_path, output_path):
    """可视化单张图片的标注问题"""
    model = YOLO(model_path)
    result = analyze_image(model, img_path, label_path)

    if not result:
        print("无法分析图片")
        return

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    # 加载标注并绘制（绿色）
    gt_labels = load_yolo_labels(label_path)
    for lbl in gt_labels:
        box = xywh_to_xyxy(lbl['x'], lbl['y'], lbl['w'], lbl['h'], img_w, img_h)
        box = [int(x) for x in box]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(img, f"GT:{lbl['class']}", (box[0], box[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 绘制漏标（红色）
    for pred in result['issues']['missing_labels']:
        box = [int(x) for x in pred['box']]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(img, f"MISS:{pred['class']} {pred['conf']:.2f}", (box[0], box[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 绘制多标（黄色）
    for gt in result['issues']['extra_labels']:
        box = xywh_to_xyxy(gt['box'][0], gt['box'][1], gt['box'][2], gt['box'][3], 1, 1)
        # 这里gt['box']已经是xyxy格式
        box = [int(x) for x in gt['box']]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        cv2.putText(img, f"EXTRA:{gt['class']}", (box[0], box[3]+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imwrite(output_path, img)
    print(f"可视化保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='数据清洗工具')
    parser.add_argument('--model', '-m', default='models/bifpn_best.pt', help='模型路径')
    parser.add_argument('--data', '-d', default='data/merged_dataset/train', help='数据目录')
    parser.add_argument('--output', '-o', default='data/cleaning_results', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--visualize', '-v', help='可视化单张图片')
    args = parser.parse_args()

    if args.visualize:
        # 可视化模式
        label_path = args.visualize.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        output_path = 'cleaning_vis.jpg'
        visualize_issues(args.model, args.visualize, label_path, output_path)
    else:
        # 清洗模式
        clean_dataset(args.model, args.data, args.output, args.conf)


if __name__ == '__main__':
    main()
