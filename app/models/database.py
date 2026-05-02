from datetime import datetime
from app import db


class Detection(db.Model):
    """检测记录"""
    __tablename__ = 'detections'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(20))  # image/video
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    violations = db.relationship('Violation', backref='detection', lazy='dynamic')


class Violation(db.Model):
    """违规记录"""
    __tablename__ = 'violations'

    id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey('detections.id'), nullable=False)
    violation_type = db.Column(db.String(50))  # passenger / no_helmet
    frame_number = db.Column(db.Integer)
    timestamp = db.Column(db.Float)  # 视频中的时间戳
    bbox = db.Column(db.JSON)  # 边界框坐标
    confidence = db.Column(db.Float)
    screenshot_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Statistics(db.Model):
    """统计数据"""
    __tablename__ = 'statistics'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    total_detections = db.Column(db.Integer, default=0)
    passenger_violations = db.Column(db.Integer, default=0)
    helmet_violations = db.Column(db.Integer, default=0)
    ebike_count = db.Column(db.Integer, default=0)
