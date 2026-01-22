"""
训练脚本: YOLOv8n + BiFPN特征融合
改进1: 增强多尺度特征融合能力
"""
import sys
sys.path.insert(0, '/home/ubuntu/yolo_zhy')

from ultralytics import YOLO
from pathlib import Path


def train_bifpn():
    """Train YOLOv8n with BiFPN-style feature fusion."""

    # 使用自定义配置
    model = YOLO('/home/ubuntu/yolo_zhy/models/yolov8n_bifpn.yaml')

    # 加载预训练权重 (backbone部分)
    model.load('yolov8n.pt')

    # 训练配置
    results = model.train(
        data='/home/ubuntu/yolo_zhy/data/merged_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=50,
        device=0,
        workers=4,
        project='runs/detect',
        name='bifpn_v1',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        plots=True,
        save=True,
        val=True,
    )

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("改进1: YOLOv8n + BiFPN特征融合")
    print("=" * 60)
    train_bifpn()
