import cv2
import numpy as np
from ultralytics import YOLO
from config import Config


class EbikeDetector:
    """电动车违规检测器"""

    def __init__(self, model_path=None):
        self.model_path = model_path or Config.MODEL_PATH
        self.model = None
        self.class_names = Config.CLASS_NAMES
        self.conf_threshold = Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = Config.IOU_THRESHOLD

    def load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.model_path)
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

    def detect_violations(self, detections):
        """
        根据检测结果判断违规行为
        Args:
            detections: YOLO检测结果
        Returns:
            violations: 违规列表
        """
        violations = []
        boxes = detections.boxes

        if boxes is None or len(boxes) == 0:
            return violations

        # 提取各类别的检测框
        ebikes = []
        drivers = []
        passengers = []
        helmets = []

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            detection_info = {
                'bbox': xyxy.tolist(),
                'confidence': conf,
                'class': cls
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
            violations.append({
                'type': 'passenger',
                'bbox': passenger['bbox'],
                'confidence': passenger['confidence'],
                'description': '电动车载人违规'
            })

        # 检测头盔违规：驾驶员/乘客未佩戴头盔
        for person in drivers + passengers:
            person_bbox = person['bbox']
            has_helmet = False

            for helmet in helmets:
                if self._is_helmet_on_person(helmet['bbox'], person_bbox):
                    has_helmet = True
                    break

            if not has_helmet:
                violations.append({
                    'type': 'no_helmet',
                    'bbox': person['bbox'],
                    'confidence': person['confidence'],
                    'description': '未佩戴头盔'
                })

        return violations

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

                color = colors.get(cls, (255, 255, 255))
                label = f"{self.class_names[cls]} {conf:.2f}"

                cv2.rectangle(result_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(result_image, label, (xyxy[0], xyxy[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 标记违规
        for v in violations:
            bbox = [int(x) for x in v['bbox']]
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            cv2.putText(result_image, v['description'], (bbox[0], bbox[3] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return result_image
