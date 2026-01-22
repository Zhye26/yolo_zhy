import cv2
import os
import numpy as np
from ultralytics import YOLO
from config import Config


class EbikeDetector:
    """电动车违规检测器"""

    def __init__(self, model_path=None, use_tensorrt=False):
        self.model_path = model_path or Config.MODEL_PATH
        self.use_tensorrt = use_tensorrt
        self.model = None
        self.class_names = Config.CLASS_NAMES
        self.conf_threshold = Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = Config.IOU_THRESHOLD
        self.tracked_violations = {}  # 记录已跟踪的违规ID

    def load_model(self):
        """加载YOLO模型"""
        try:
            model_path = self.model_path
            # 如果启用TensorRT，尝试加载.engine文件
            if self.use_tensorrt:
                engine_path = model_path.replace('.pt', '.engine')
                if os.path.exists(engine_path):
                    model_path = engine_path
                    print(f"使用TensorRT模型: {engine_path}")
                else:
                    print(f"TensorRT模型不存在，使用PyTorch: {model_path}")

            self.model = YOLO(model_path)
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            # 使用预训练模型作为后备
            self.model = YOLO('yolov8n.pt')
            return False

    def detect(self, image):
        """
        对单张图片进行检测
        Args:
            image: numpy array (BGR format)
        Returns:
            results: 检测结果
        """
        if self.model is None:
            self.load_model()

        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        return results[0]

    def track(self, image, persist=True):
        """
        对图片进行检测+跟踪 (ByteTrack)
        Args:
            image: numpy array (BGR format)
            persist: 是否保持跟踪状态
        Returns:
            results: 带跟踪ID的检测结果
        """
        if self.model is None:
            self.load_model()

        results = self.model.track(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=persist,
            tracker="bytetrack.yaml"
        )
        return results[0]

    def reset_tracker(self):
        """重置跟踪器状态"""
        self.tracked_violations = {}
        if self.model is not None:
            self.model.predictor = None

    def detect_violations(self, detections, use_tracking=False):
        """
        根据检测结果判断违规行为
        Args:
            detections: YOLO检测结果
            use_tracking: 是否使用跟踪ID去重
        Returns:
            violations: 违规列表
            new_violations: 新发现的违规（去重后）
        """
        violations = []
        new_violations = []
        boxes = detections.boxes

        if boxes is None or len(boxes) == 0:
            return violations, new_violations

        # 提取各类别的检测框
        ebikes = []
        drivers = []
        passengers = []
        helmets = []

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            track_id = int(box.id[0]) if box.id is not None else None

            detection_info = {
                'bbox': xyxy.tolist(),
                'confidence': conf,
                'class': cls,
                'track_id': track_id
            }

            if cls == 0:  # ebike
                ebikes.append(detection_info)
            elif cls == 1:  # driver
                drivers.append(detection_info)
            elif cls == 2:  # passenger
                passengers.append(detection_info)
            elif cls == 3:  # helmet
                helmets.append(detection_info)

        # 检测载人违规：电动车区域内有乘客
        for passenger in passengers:
            violation = {
                'type': 'passenger',
                'bbox': passenger['bbox'],
                'confidence': passenger['confidence'],
                'track_id': passenger['track_id'],
                'description': '电动车载人违规'
            }
            violations.append(violation)

            # 去重：检查是否是新违规
            if use_tracking and passenger['track_id'] is not None:
                vid = f"passenger_{passenger['track_id']}"
                if vid not in self.tracked_violations:
                    self.tracked_violations[vid] = True
                    new_violations.append(violation)
            else:
                new_violations.append(violation)

        # 检测头盔违规：驾驶员/乘客未佩戴头盔
        for person in drivers + passengers:
            person_bbox = person['bbox']
            has_helmet = False

            for helmet in helmets:
                if self._is_helmet_on_person(helmet['bbox'], person_bbox):
                    has_helmet = True
                    break

            if not has_helmet:
                violation = {
                    'type': 'no_helmet',
                    'bbox': person['bbox'],
                    'confidence': person['confidence'],
                    'track_id': person['track_id'],
                    'description': '未佩戴头盔'
                }
                violations.append(violation)

                # 去重
                if use_tracking and person['track_id'] is not None:
                    vid = f"no_helmet_{person['track_id']}"
                    if vid not in self.tracked_violations:
                        self.tracked_violations[vid] = True
                        new_violations.append(violation)
                else:
                    new_violations.append(violation)

        return violations, new_violations

    def _is_helmet_on_person(self, helmet_bbox, person_bbox, threshold=0.3):
        """判断头盔是否在人的头部区域"""
        hx1, hy1, hx2, hy2 = helmet_bbox
        px1, py1, px2, py2 = person_bbox

        # 头部区域：人体框上部1/3
        head_y2 = py1 + (py2 - py1) * 0.35

        # 检查头盔是否在头部区域
        if hy2 > head_y2:
            return False

        # 检查水平重叠
        overlap_x = max(0, min(hx2, px2) - max(hx1, px1))
        helmet_width = hx2 - hx1

        if helmet_width > 0 and overlap_x / helmet_width > threshold:
            return True

        return False

    def draw_results(self, image, detections, violations):
        """在图片上绘制检测结果"""
        result_image = image.copy()
        boxes = detections.boxes

        colors = {
            0: (0, 255, 0),    # ebike - green
            1: (255, 0, 0),    # driver - blue
            2: (0, 165, 255),  # passenger - orange
            3: (255, 255, 0)   # helmet - cyan
        }

        # 绘制检测框
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box.id[0]) if box.id is not None else None

                color = colors.get(cls, (255, 255, 255))
                label = f"{self.class_names[cls]} {conf:.2f}"
                if track_id is not None:
                    label = f"ID:{track_id} {label}"

                cv2.rectangle(result_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(result_image, label, (xyxy[0], xyxy[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 标记违规
        for v in violations:
            bbox = [int(x) for x in v['bbox']]
            desc = v['description']
            if v.get('track_id') is not None:
                desc = f"ID:{v['track_id']} {desc}"
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            cv2.putText(result_image, desc, (bbox[0], bbox[3] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return result_image
