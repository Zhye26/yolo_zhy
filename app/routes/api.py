from flask import Blueprint, jsonify, request, Response
from app.models import Detection, Violation, Statistics
from app.services import EbikeDetector, VideoProcessor
from app import db
from sqlalchemy import func
from datetime import datetime, timedelta

bp = Blueprint('api', __name__)

detector = EbikeDetector()


@bp.route('/stats')
def get_stats():
    """获取统计数据"""
    today = datetime.utcnow().date()

    stats = {
        'total_detections': Detection.query.count(),
        'today_detections': Detection.query.filter(
            func.date(Detection.created_at) == today
        ).count(),
        'passenger_violations': Violation.query.filter_by(violation_type='passenger').count(),
        'helmet_violations': Violation.query.filter_by(violation_type='no_helmet').count(),
    }

    return jsonify(stats)


@bp.route('/violations')
def get_violations():
    """获取违规记录列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    violation_type = request.args.get('type')

    query = Violation.query

    if violation_type:
        query = query.filter_by(violation_type=violation_type)

    pagination = query.order_by(Violation.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    violations = [{
        'id': v.id,
        'type': v.violation_type,
        'confidence': v.confidence,
        'created_at': v.created_at.isoformat(),
        'detection_id': v.detection_id
    } for v in pagination.items]

    return jsonify({
        'violations': violations,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })


@bp.route('/chart/daily')
def daily_chart():
    """获取每日统计图表数据"""
    days = request.args.get('days', 7, type=int)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    # 按日期分组统计
    results = db.session.query(
        func.date(Violation.created_at).label('date'),
        Violation.violation_type,
        func.count(Violation.id).label('count')
    ).filter(
        func.date(Violation.created_at) >= start_date
    ).group_by(
        func.date(Violation.created_at),
        Violation.violation_type
    ).all()

    # 整理数据
    data = {}
    for r in results:
        date_str = str(r.date)
        if date_str not in data:
            data[date_str] = {'passenger': 0, 'no_helmet': 0}
        data[date_str][r.violation_type] = r.count

    return jsonify(data)


@bp.route('/stream')
def video_stream():
    """实时视频流检测"""
    stream_url = request.args.get('url', 0)  # 默认使用摄像头

    processor = VideoProcessor(detector)
    return Response(
        processor.process_stream(stream_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
