"""
数据集整合脚本
将头盔数据集和电动车数据集合并为统一的违规检测数据集
"""

import os
import shutil
from pathlib import Path


# 目标类别映射
# 0: ebike (电动车)
# 1: driver (驾驶员) - 暂时用person代替，后续可手动区分
# 2: passenger (乘客) - 需要手动标注
# 3: helmet (头盔)

def merge_datasets():
    base_dir = Path('/home/ubuntu/yolo_zhy/data')
    output_dir = base_dir / 'merged_dataset'

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 数据集1: 电动车数据集
    # 原类别: 0=Electric-bicycle, 1=person
    # 映射: 0->0(ebike), 1->1(driver)
    ebike_dir = base_dir / 'ebike_dataset'
    ebike_mapping = {0: 0, 1: 1}  # Electric-bicycle->ebike, person->driver

    # 数据集2: 头盔数据集
    # 原类别: 0=full, 1=half, 2=invalid, 3=not_wearing
    # 映射: 0,1,2->3(helmet), 3->忽略(不是头盔)
    helmet_dir = base_dir / 'helmet_dataset'

    split_mapping = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }

    # 处理电动车数据集
    print("处理电动车数据集...")
    for src_split, dst_split in split_mapping.items():
        img_src = ebike_dir / src_split / 'images'
        lbl_src = ebike_dir / src_split / 'labels'

        if not img_src.exists():
            continue

        for img_file in img_src.glob('*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            # 复制图片
            new_name = f"ebike_{img_file.name}"
            shutil.copy(img_file, output_dir / 'images' / dst_split / new_name)

            # 转换标签
            lbl_file = lbl_src / f"{img_file.stem}.txt"
            if lbl_file.exists():
                new_labels = []
                with open(lbl_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_cls = int(parts[0])
                            new_cls = ebike_mapping.get(old_cls, old_cls)
                            new_labels.append(f"{new_cls} {' '.join(parts[1:])}")

                with open(output_dir / 'labels' / dst_split / f"ebike_{img_file.stem}.txt", 'w') as f:
                    f.write('\n'.join(new_labels))

    # 处理头盔数据集
    print("处理头盔数据集...")
    for src_split, dst_split in split_mapping.items():
        img_src = helmet_dir / src_split / 'images'
        lbl_src = helmet_dir / src_split / 'labels'

        if not img_src.exists():
            continue

        for img_file in img_src.glob('*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            # 复制图片
            new_name = f"helmet_{img_file.name}"
            shutil.copy(img_file, output_dir / 'images' / dst_split / new_name)

            # 转换标签 (full/half/invalid -> helmet, not_wearing -> 忽略)
            lbl_file = lbl_src / f"{img_file.stem}.txt"
            if lbl_file.exists():
                new_labels = []
                with open(lbl_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_cls = int(parts[0])
                            # 0,1,2 (full/half/invalid) -> 3 (helmet)
                            # 3 (not_wearing) -> 跳过
                            if old_cls in [0, 1, 2]:
                                new_labels.append(f"3 {' '.join(parts[1:])}")

                with open(output_dir / 'labels' / dst_split / f"helmet_{img_file.stem}.txt", 'w') as f:
                    f.write('\n'.join(new_labels))

    # 统计
    print("\n合并完成! 统计:")
    for split in ['train', 'val', 'test']:
        img_count = len(list((output_dir / 'images' / split).glob('*')))
        print(f"  {split}: {img_count} 张图片")


if __name__ == '__main__':
    merge_datasets()
