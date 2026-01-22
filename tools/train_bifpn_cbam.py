"""
训练脚本: YOLOv8n + BiFPN + CBAM注意力机制
改进2: 在BiFPN基础上添加CBAM，增强对头盔等小目标的关注
"""
import sys
sys.path.insert(0, '/home/ubuntu/yolo_zhy')

import torch
from pathlib import Path

# Register CBAM module BEFORE importing YOLO
from models.custom_modules.cbam import CBAM, ChannelAttention, SpatialAttention, C2f_CBAM

# Register to ultralytics.nn.tasks globals
import ultralytics.nn.tasks as tasks
tasks.CBAM = CBAM

# Also register to modules
from ultralytics.nn.modules import conv, block
conv.CBAM = CBAM
block.CBAM = CBAM

# Now import YOLO
from ultralytics import YOLO


def train_bifpn_cbam():
    """Train YOLOv8n with BiFPN + CBAM attention."""

    # Use custom config
    model = YOLO('/home/ubuntu/yolo_zhy/models/yolov8n_bifpn_cbam.yaml')

    # Load pretrained BiFPN weights
    pretrained = torch.load('/home/ubuntu/yolo_zhy/models/bifpn_best.pt', map_location='cpu', weights_only=False)
    model_state = model.model.state_dict()

    # Transfer compatible weights
    pretrained_state = pretrained['model'].state_dict() if hasattr(pretrained['model'], 'state_dict') else pretrained['model'].float().state_dict()

    transferred = 0
    for k, v in pretrained_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            transferred += 1

    model.model.load_state_dict(model_state, strict=False)
    print(f"Transferred {transferred}/{len(pretrained_state)} items from BiFPN pretrained weights")

    # Training config
    results = model.train(
        data='/home/ubuntu/yolo_zhy/data/merged_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=50,
        device=0,
        workers=4,
        project='runs/detect',
        name='bifpn_cbam_v1',
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
    )

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("改进2: YOLOv8n + BiFPN + CBAM注意力机制")
    print("=" * 60)
    train_bifpn_cbam()
