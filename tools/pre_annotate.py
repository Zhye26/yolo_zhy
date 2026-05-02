"""
预标注脚本 - 使用COCO预训练模型生成初始标注
用于加速数据集标注过程
"""

import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


# COCO类别到我们类别的映射
# COCO: person=0, bicycle=1, car=2, motorcycle=3
# 我们: ebike=0, driver=1, passenger=2, helmet=3
COCO_TO_CUSTOM = {
    3: 0,   # motorcycle -> ebike
    1: 0,   # bicycle -> ebike (电动车外形类似)
    0: 1,   # person -> driver (默认标为driver，后续手动区分passenger)
}


def pre_annotate(image_dir, output_dir, model_path='yolov8n.pt', conf=0.3):
    """
    对图片进行预标注

    Args:
        image_dir: 图片目录
        output_dir: 标注输出目录
        model_path: YOLO模型路径
        conf: 置信度阈值
    """
    # 加载模型
    model = YOLO(model_path)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # 获取所有图片
    image_dir = Path(image_dir)
    images = [f for f in image_dir.iterdir()
              if f.suffix.lower() in image_extensions]

    print(f"找到 {len(images)} 张图片")

    for img_path in images:
        # 读取图片获取尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取: {img_path}")
            continue

        h, w = img.shape[:2]

        # 检测
        results = model(img, conf=conf, verbose=False)[0]

        # 生成标注
        annotations = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])

                # 只处理我们关心的类别
                if cls not in COCO_TO_CUSTOM:
                    continue

                custom_cls = COCO_TO_CUSTOM[cls]

                # 获取边界框 (xyxy格式)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # 转换为YOLO格式 (归一化的 x_center, y_center, width, height)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bbox_w = (x2 - x1) / w
                bbox_h = (y2 - y1) / h

                annotations.append(f"{custom_cls} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")

        # 保存标注文件
        label_path = Path(output_dir) / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))

        print(f"✓ {img_path.name} -> {len(annotations)} 个目标")

    print(f"\n预标注完成! 标注文件保存在: {output_dir}")
    print("\n下一步:")
    print("1. 使用 LabelImg 打开图片目录进行修正")
    print("2. 区分 driver(1) 和 passenger(2)")
    print("3. 手动添加 helmet(3) 标注")


def main():
    parser = argparse.ArgumentParser(description='预标注工具')
    parser.add_argument('--images', type=str, required=True, help='图片目录')
    parser.add_argument('--output', type=str, required=True, help='标注输出目录')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO模型')
    parser.add_argument('--conf', type=float, default=0.3, help='置信度阈值')

    args = parser.parse_args()
    pre_annotate(args.images, args.output, args.model, args.conf)


if __name__ == '__main__':
    main()
