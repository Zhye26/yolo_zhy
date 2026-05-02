"""
视频抽帧脚本 - 从视频中提取帧用于标注
"""

import os
import cv2
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, interval=30, max_frames=None):
    """
    从视频中提取帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        interval: 抽帧间隔（每隔多少帧提取一帧）
        max_frames: 最大提取帧数
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {total_frames} 帧, {fps:.1f} FPS")
    print(f"抽帧间隔: 每 {interval} 帧 (约 {interval/fps:.1f} 秒)")

    video_name = Path(video_path).stem
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            output_path = os.path.join(output_dir, f"{video_name}_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

            if max_frames and saved_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    print(f"\n提取完成! 共保存 {saved_count} 帧到: {output_dir}")


def batch_extract(video_dir, output_dir, interval=30, max_per_video=100):
    """批量处理多个视频"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

    video_dir = Path(video_dir)
    videos = [f for f in video_dir.iterdir()
              if f.suffix.lower() in video_extensions]

    print(f"找到 {len(videos)} 个视频文件")

    for video_path in videos:
        print(f"\n处理: {video_path.name}")
        extract_frames(str(video_path), output_dir, interval, max_per_video)


def main():
    parser = argparse.ArgumentParser(description='视频抽帧工具')
    parser.add_argument('--video', type=str, help='单个视频文件')
    parser.add_argument('--video-dir', type=str, help='视频目录（批量处理）')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--interval', type=int, default=30, help='抽帧间隔')
    parser.add_argument('--max-frames', type=int, default=None, help='最大帧数')

    args = parser.parse_args()

    if args.video:
        extract_frames(args.video, args.output, args.interval, args.max_frames)
    elif args.video_dir:
        batch_extract(args.video_dir, args.output, args.interval, args.max_frames or 100)
    else:
        print("请指定 --video 或 --video-dir")


if __name__ == '__main__':
    main()
