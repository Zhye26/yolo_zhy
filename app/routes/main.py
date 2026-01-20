from flask import Blueprint, render_template
from app.models import Detection, Violation
from app import db
from sqlalchemy import func
from datetime import datetime, timedelta

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    """首页 - 仪表板"""
    # 统计数据
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)

    stats = {
        'total_detections': Detection.query.count(),
        'today_detections': Detection.query.filter(
            func.date(Detection.created_at) == today
        ).count(),
        'passenger_violations': Violation.query.filter_by(violation_type='passenger').count(),
        'helmet_violations': Violation.query.filter_by(violation_type='no_helmet').count(),
    }

    # 最近违规记录
    recent_violations = Violation.query.order_by(
        Violation.created_at.desc()
    ).limit(10).all()

    return render_template('index.html', stats=stats, recent_violations=recent_violations)


@bp.route('/history')
def history():
    """历史记录页面"""
    detections = Detection.query.order_by(Detection.created_at.desc()).all()
    return render_template('history.html', detections=detections)


@bp.route('/statistics')
def statistics():
    """统计分析页面"""
    return render_template('statistics.html')
