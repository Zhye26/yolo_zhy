"""
数据集划分脚本 - 将标注好的数据划分为训练集/验证集/测试集
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    """
    划分数据集

    Args:
        image_dir: 图片目录
        label_dir: 标注目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    test_ratio = 1 - train_ratio - val_ratio

    # 获取所有有标注的图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    images = []
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                images.append((img_path, label_path))

    print(f"找到 {len(images)} 对图片-标注文件")

    # 随机打乱
    random.shuffle(images)

    # 划分
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    # 创建目录并复制文件
    output_dir = Path(output_dir)
    for split_name, split_data in splits.items():
        img_out = output_dir / 'images' / split_name
        lbl_out = output_dir / 'labels' / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_data:
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(label_path, lbl_out / label_path.name)

        print(f"{split_name}: {len(split_data)} 张")

    print(f"\n数据集划分完成! 保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='数据集划分工具')
    parser.add_argument('--images', type=str, required=True, help='图片目录')
    parser.add_argument('--labels', type=str, required=True, help='标注目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--train', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val', type=float, default=0.2, help='验证集比例')

    args = parser.parse_args()
    split_dataset(args.images, args.labels, args.output, args.train, args.val)


if __name__ == '__main__':
    main()
