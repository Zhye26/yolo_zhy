#!/usr/bin/env python3
"""
ByteTrack跟踪测试脚本
用于验证跟踪功能和违规去重效果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from app.services import EbikeDetector, YoloSam3Detector
from app.services.video_processor import VideoProcessor


def build_detector(args):
    """Build detector based on CLI args."""
    if args.sam3:
        print(f"SAM3权重: {args.sam3_checkpoint}")
        print(f"SAM3顺序加载: {'开启' if args.sam3_sequential else '关闭'}")
        detector = YoloSam3Detector(
            model_path=args.model or 'models/bifpn_best.pt',
            sam3_checkpoint=args.sam3_checkpoint,
            mask_alpha=args.sam3_alpha,
            sequential_mode=args.sam3_sequential,
        )
    else:
        detector = EbikeDetector(model_path=args.model or 'models/bifpn_best.pt')

    detector.load_model()
    return detector


def test_tracking(video_path, args, output_path=None):
    """测试跟踪功能"""
    print(f"测试视频: {video_path}")
    print(f"模型路径: {args.model or 'models/bifpn_best.pt'}")
    print(f"分割模式: {'YOLO+SAM3' if args.sam3 else '仅YOLO检测'}")
    print("-" * 50)

    detector = build_detector(args)
    processor = VideoProcessor(detector)

    print("\n[1] 使用ByteTrack跟踪处理视频...")
    result_tracking = processor.process_video(
        video_path,
        output_path=output_path,
        use_tracking=True,
        callback=lambda progress: print(f"  进度: {progress:.1f}%", end='\r')
    )
    print()

    print("\n[2] 不使用跟踪处理视频...")
    detector.reset_tracker()
    result_no_tracking = processor.process_video(
        video_path,
        output_path=output_path.replace('.mp4', '_no_track.mp4') if output_path else None,
        use_tracking=False,
        callback=lambda progress: print(f"  进度: {progress:.1f}%", end='\r')
    )
    print()

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

    print("\n按违规类型统计:")
    print("-" * 40)

    types_no_track = {}
    for violation in result_no_tracking['violations']:
        violation_type = violation['type']
        types_no_track[violation_type] = types_no_track.get(violation_type, 0) + 1

    types_track = {}
    for violation in result_tracking['unique_violations']:
        violation_type = violation['type']
        types_track[violation_type] = types_track.get(violation_type, 0) + 1

    all_types = set(types_no_track.keys()) | set(types_track.keys())
    for violation_type in all_types:
        no_track_total = types_no_track.get(violation_type, 0)
        track_total = types_track.get(violation_type, 0)
        print(f"  {violation_type}: {no_track_total} -> {track_total} (减少 {no_track_total - track_total})")

    print(f"\n输出视频: {result_tracking['output_path']}")
    return result_tracking, result_no_tracking


def main():
    parser = argparse.ArgumentParser(description='ByteTrack跟踪测试')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('--model', '-m', default='models/bifpn_best.pt', help='模型路径')
    parser.add_argument('--output', '-o', help='输出视频路径')
    parser.add_argument('--sam3', action='store_true', help='启用SAM3分割叠加')
    parser.add_argument('--sam3-checkpoint', default='/home/ubuntu/SAM3/sam3.pt', help='SAM3权重路径')
    parser.add_argument('--sam3-alpha', type=float, default=0.35, help='SAM3掩码透明度')
    parser.add_argument('--sam3-sequential', action='store_true', help='按需加载SAM3，降低显存占用')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        sys.exit(1)

    test_tracking(args.video, args, args.output)


if __name__ == '__main__':
    main()
