"""
训练脚本: YOLOv8n + BiFPN + SimAM注意力机制
改进3: 使用零参数SimAM替代CBAM，避免与BiFPN冲突

SimAM优势:
- 零参数: 不增加模型复杂度
- 基于能量函数: 不与BiFPN的学习权重冲突
- 轻量高效: 适合YOLOv8n等小模型
"""
import sys
sys.path.insert(0, '/home/ubuntu/yolo_zhy')

import torch
from pathlib import Path

# Register SimAM module BEFORE importing YOLO
from models.custom_modules.simam import SimAM, SimAM_YOLOv8

# Register to ultralytics.nn.tasks globals
import ultralytics.nn.tasks as tasks
tasks.SimAM = SimAM

# Also register to modules
from ultralytics.nn.modules import conv, block
conv.SimAM = SimAM
block.SimAM = SimAM

# Now import YOLO
from ultralytics import YOLO


def train_bifpn_simam():
    """Train YOLOv8n with BiFPN + SimAM attention."""

    # Use custom config
    model = YOLO('/home/ubuntu/yolo_zhy/models/yolov8n_bifpn_simam.yaml')

    # Load pretrained BiFPN weights
    pretrained_path = '/home/ubuntu/yolo_zhy/models/bifpn_best.pt'
    if Path(pretrained_path).exists():
        pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        model_state = model.model.state_dict()

        # Transfer compatible weights
        if hasattr(pretrained['model'], 'state_dict'):
            pretrained_state = pretrained['model'].state_dict()
        else:
            pretrained_state = pretrained['model'].float().state_dict()

        transferred = 0
        for k, v in pretrained_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                transferred += 1

        model.model.load_state_dict(model_state, strict=False)
        print(f"Transferred {transferred}/{len(pretrained_state)} weights from BiFPN")
    else:
        print("BiFPN weights not found, training from scratch")
        model.load('yolov8n.pt')

    # Training config with improvements from analysis
    results = model.train(
        data='/home/ubuntu/yolo_zhy/data/merged_dataset/data.yaml',
        epochs=100,
        imgsz=640,  # Can increase to 960 for better small object detection
        batch=16,
        patience=50,
        device=0,
        workers=4,
        project='runs/detect',
        name='bifpn_simam_v1',
        exist_ok=True,
        pretrained=False,  # Already loaded custom weights
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
        amp=True,  # Mixed precision training
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,  # Enable for small object boost: 0.3
    )

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("改进3: YOLOv8n + BiFPN + SimAM注意力机制")
    print("=" * 60)
    train_bifpn_simam()
