#!/usr/bin/env python3
"""
TensorRT推理速度测试脚本
对比PyTorch和TensorRT模型的推理速度
"""
import argparse
import os
import time
import numpy as np
from ultralytics import YOLO


def benchmark_model(model_path, imgsz=640, warmup=10, runs=100):
    """
    测试模型推理速度
    Args:
        model_path: 模型路径
        imgsz: 输入图像尺寸
        warmup: 预热次数
        runs: 测试次数
    Returns:
        dict: 测试结果
    """
    print(f"\n测试模型: {model_path}")
    model = YOLO(model_path)

    # 生成随机测试图像
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # 预热
    print(f"预热 {warmup} 次...")
    for _ in range(warmup):
        model(dummy_input, verbose=False)

    # 测试
    print(f"测试 {runs} 次...")
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model(dummy_input, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times = np.array(times)
    results = {
        'model': os.path.basename(model_path),
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000 / np.mean(times)
    }

    print(f"  平均: {results['mean_ms']:.2f}ms (±{results['std_ms']:.2f})")
    print(f"  FPS: {results['fps']:.1f}")

    return results


def compare_models(pt_path, engine_path, imgsz=640):
    """对比PyTorch和TensorRT模型"""
    print("=" * 60)
    print("TensorRT推理速度对比测试")
    print("=" * 60)

    # 测试PyTorch模型
    pt_results = benchmark_model(pt_path, imgsz)

    # 测试TensorRT模型
    trt_results = benchmark_model(engine_path, imgsz)

    # 对比结果
    speedup = pt_results['mean_ms'] / trt_results['mean_ms']

    print("\n" + "=" * 60)
    print("对比结果:")
    print("=" * 60)
    print(f"{'指标':<15} {'PyTorch':<15} {'TensorRT':<15} {'提升'}")
    print("-" * 60)
    print(f"{'延迟(ms)':<15} {pt_results['mean_ms']:<15.2f} {trt_results['mean_ms']:<15.2f} {speedup:.2f}x")
    print(f"{'FPS':<15} {pt_results['fps']:<15.1f} {trt_results['fps']:<15.1f} {speedup:.2f}x")

    # 获取模型大小
    pt_size = os.path.getsize(pt_path) / (1024 * 1024)
    trt_size = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"{'模型大小(MB)':<15} {pt_size:<15.1f} {trt_size:<15.1f} -")

    return {
        'pytorch': pt_results,
        'tensorrt': trt_results,
        'speedup': speedup
    }


def main():
    parser = argparse.ArgumentParser(description='TensorRT推理速度测试')
    parser.add_argument('--pt', default='models/bifpn_best.pt', help='PyTorch模型路径')
    parser.add_argument('--engine', help='TensorRT模型路径（默认自动推断）')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    parser.add_argument('--runs', type=int, default=100, help='测试次数')
    args = parser.parse_args()

    if not os.path.exists(args.pt):
        print(f"错误: PyTorch模型不存在: {args.pt}")
        return

    # 自动推断TensorRT模型路径
    engine_path = args.engine
    if engine_path is None:
        engine_path = args.pt.replace('.pt', '.engine')

    if not os.path.exists(engine_path):
        print(f"错误: TensorRT模型不存在: {engine_path}")
        print(f"请先运行: python tools/export_tensorrt.py {args.pt}")
        return

    compare_models(args.pt, engine_path, args.imgsz)


if __name__ == '__main__':
    main()
