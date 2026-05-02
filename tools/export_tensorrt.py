#!/usr/bin/env python3
"""
TensorRT模型导出脚本
将YOLO模型导出为TensorRT格式，支持FP16量化
"""
import argparse
import os
import time
from ultralytics import YOLO


def export_tensorrt(model_path, half=True, imgsz=640, batch=1):
    """
    导出模型为TensorRT格式
    Args:
        model_path: PyTorch模型路径
        half: 是否使用FP16
        imgsz: 输入图像尺寸
        batch: 批处理大小
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"\n导出配置:")
    print(f"  - 精度: {'FP16' if half else 'FP32'}")
    print(f"  - 图像尺寸: {imgsz}")
    print(f"  - 批处理: {batch}")

    start = time.time()
    export_path = model.export(
        format="engine",
        half=half,
        imgsz=imgsz,
        batch=batch,
        device=0,
        simplify=True,
        workspace=4  # GB
    )
    elapsed = time.time() - start

    print(f"\n导出完成!")
    print(f"  - 耗时: {elapsed:.1f}秒")
    print(f"  - 输出: {export_path}")

    # 获取文件大小
    if os.path.exists(export_path):
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        print(f"  - 大小: {size_mb:.1f}MB")

    return export_path


def main():
    parser = argparse.ArgumentParser(description='导出YOLO模型为TensorRT格式')
    parser.add_argument('model', help='PyTorch模型路径 (.pt)')
    parser.add_argument('--fp32', action='store_true', help='使用FP32精度（默认FP16）')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=1, help='批处理大小')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return

    export_tensorrt(
        args.model,
        half=not args.fp32,
        imgsz=args.imgsz,
        batch=args.batch
    )


if __name__ == '__main__':
    main()
