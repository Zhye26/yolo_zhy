#!/usr/bin/env python3
"""
ByteTrack跟踪测试脚本
用于验证跟踪功能和违规去重效果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import argparse
from app.services.detector import EbikeDetector
from app.services.video_processor import VideoProcessor


def test_tracking(video_path, model_path=None, output_path=None):
    """测试跟踪功能"""
    print(f"测试视频: {video_path}")
    print(f"模型路径: {model_path or 'models/bifpn_best.pt'}")
    print("-" * 50)

    # 初始化检测器
    detector = EbikeDetector(model_path=model_path or "models/bifpn_best.pt")
    detector.load_model()

    processor = VideoProcessor(detector)

    # 处理视频（带跟踪）
    print("\n[1] 使用ByteTrack跟踪处理视频...")
    result_tracking = processor.process_video(
        video_path,
        output_path=output_path,
        use_tracking=True,
        callback=lambda p: print(f"  进度: {p:.1f}%", end='\r')
    )
    print()

    # 处理视频（不带跟踪）
    print("\n[2] 不使用跟踪处理视频...")
    detector.reset_tracker()
    result_no_tracking = processor.process_video(
        video_path,
        output_path=output_path.replace('.mp4', '_no_track.mp4') if output_path else None,
        use_tracking=False,
        callback=lambda p: print(f"  进度: {p:.1f}%", end='\r')
    )
    print()

    # 对比结果
    print("\n" + "=" * 50)
    print("对比结果:")
    print("=" * 50)
    print(f"总帧数: {result_tracking['total_frames']}")
    print(f"FPS: {result_tracking['fps']}")
    print()
    print(f"{'指标':<20} {'无跟踪':<15} {'有跟踪':<15} {'减少'}")
    print("-" * 60)

    no_track_count = len(result_no_tracking['violations'])
    track_all_count = len(result_tracking['violations'])
    track_unique_count = len(result_tracking['unique_violations'])

    print(f"{'总违规检测次数':<20} {no_track_count:<15} {track_all_count:<15} -")
    print(f"{'唯一违规数量':<20} {no_track_count:<15} {track_unique_count:<15} {no_track_count - track_unique_count}")

    if no_track_count > 0:
        reduction = (1 - track_unique_count / no_track_count) * 100
        print(f"\n去重效果: 减少 {reduction:.1f}% 重复计数")

    # 按类型统计
    print("\n按违规类型统计:")
    print("-" * 40)

    types_no_track = {}
    for v in result_no_tracking['violations']:
        t = v['type']
        types_no_track[t] = types_no_track.get(t, 0) + 1

    types_track = {}
    for v in result_tracking['unique_violations']:
        t = v['type']
        types_track[t] = types_track.get(t, 0) + 1

    all_types = set(types_no_track.keys()) | set(types_track.keys())
    for t in all_types:
        nt = types_no_track.get(t, 0)
        tr = types_track.get(t, 0)
        print(f"  {t}: {nt} -> {tr} (减少 {nt - tr})")

    print(f"\n输出视频: {result_tracking['output_path']}")

    return result_tracking, result_no_tracking


def main():
    parser = argparse.ArgumentParser(description='ByteTrack跟踪测试')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('--model', '-m', default='models/bifpn_best.pt', help='模型路径')
    parser.add_argument('--output', '-o', help='输出视频路径')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        sys.exit(1)

    test_tracking(args.video, args.model, args.output)


if __name__ == '__main__':
    main()
