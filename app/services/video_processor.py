import cv2
import os
from config import Config


class VideoProcessor:
    """视频流处理器"""

    def __init__(self, detector):
        self.detector = detector
        self.output_folder = Config.OUTPUT_FOLDER

    def process_video(self, video_path, output_path=None, callback=None, use_tracking=True):
        """
        处理视频文件
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            callback: 进度回调函数
            use_tracking: 是否使用ByteTrack跟踪
        Returns:
            all_violations: 所有违规记录
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            basename = os.path.basename(video_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(self.output_folder, f"{name}_result{ext}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 重置跟踪器
        if use_tracking:
            self.detector.reset_tracker()

        all_violations = []
        unique_violations = []
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测（带跟踪或不带）
            if use_tracking:
                detections = self.detector.track(frame)
            else:
                detections = self.detector.detect(frame)

            violations, new_violations = self.detector.detect_violations(
                detections, use_tracking=use_tracking
            )

            # 记录所有违规（当前帧）
            for v in violations:
                v['frame_number'] = frame_number
                v['timestamp'] = frame_number / fps
                all_violations.append(v)

            # 记录唯一违规（去重后）
            for v in new_violations:
                v['frame_number'] = frame_number
                v['timestamp'] = frame_number / fps
                unique_violations.append(v)

            # 绘制结果
            result_frame = self.detector.draw_results(frame, detections, violations)
            out.write(result_frame)

            frame_number += 1

            if callback and frame_number % 30 == 0:
                progress = frame_number / total_frames * 100
                callback(progress)

        cap.release()
        out.release()

        return {
            'output_path': output_path,
            'violations': all_violations,
            'unique_violations': unique_violations,
            'total_frames': total_frames,
            'fps': fps,
            'tracking_enabled': use_tracking
        }

    def process_stream(self, stream_url, use_tracking=True):
        """处理实时视频流（生成器）"""
        cap = cv2.VideoCapture(stream_url)

        if use_tracking:
            self.detector.reset_tracker()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if use_tracking:
                detections = self.detector.track(frame)
            else:
                detections = self.detector.detect(frame)

            violations, _ = self.detector.detect_violations(
                detections, use_tracking=use_tracking
            )
            result_frame = self.detector.draw_results(frame, detections, violations)

            # 编码为JPEG
            _, buffer = cv2.imencode('.jpg', result_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()
