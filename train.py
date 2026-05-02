"""
YOLO模型训练脚本
用于训练电动车违规检测模型
"""

import os
import argparse
from ultralytics import YOLO


def train(args):
    """训练模型"""
    # 加载预训练模型
    model = YOLO(args.model)

    # 训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project='runs/train',
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        patience=50,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    print(f"\n训练完成! 最佳模型保存在: runs/train/{args.name}/weights/best.pt")
    return results


def validate(args):
    """验证模型"""
    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    return results


def export_model(args):
    """导出模型"""
    model = YOLO(args.weights)

    # 导出为不同格式
    if args.format == 'onnx':
        model.export(format='onnx', imgsz=args.imgsz, simplify=True)
    elif args.format == 'tensorrt':
        model.export(format='engine', imgsz=args.imgsz, half=True)
    elif args.format == 'torchscript':
        model.export(format='torchscript', imgsz=args.imgsz)

    print(f"模型已导出为 {args.format} 格式")


def main():
    parser = argparse.ArgumentParser(description='电动车违规检测模型训练')
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--model', type=str, default='yolov8n.pt', help='预训练模型')
    train_parser.add_argument('--data', type=str, default='data/dataset.yaml', help='数据集配置')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    train_parser.add_argument('--batch', type=int, default=16, help='批次大小')
    train_parser.add_argument('--device', type=str, default='0', help='设备 (0, 1, cpu)')
    train_parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    train_parser.add_argument('--name', type=str, default='ebike_detection', help='实验名称')

    # 验证命令
    val_parser = subparsers.add_parser('val', help='验证模型')
    val_parser.add_argument('--weights', type=str, required=True, help='模型权重')
    val_parser.add_argument('--data', type=str, default='data/dataset.yaml', help='数据集配置')
    val_parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    val_parser.add_argument('--batch', type=int, default=16, help='批次大小')
    val_parser.add_argument('--device', type=str, default='0', help='设备')

    # 导出命令
    export_parser = subparsers.add_parser('export', help='导出模型')
    export_parser.add_argument('--weights', type=str, required=True, help='模型权重')
    export_parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'tensorrt', 'torchscript'])
    export_parser.add_argument('--imgsz', type=int, default=640, help='图像大小')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'val':
        validate(args)
    elif args.command == 'export':
        export_model(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
