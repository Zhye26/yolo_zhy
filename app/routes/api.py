from flask import Blueprint, jsonify, request, Response, current_app
from app.models import Detection, Violation, Statistics
from app.services import EbikeDetector, VideoProcessor, session_manager, SessionState
from app import db
from app.config import settings
from sqlalchemy import func, text
from datetime import datetime, timedelta
import os
import time
import cv2

bp = Blueprint('api', __name__)

detector = EbikeDetector()


@bp.route('/healthz')
def health_check():
    """Application health check endpoint."""
    model_path = settings.model.model_path

    checks = {
        'model': {
            'enabled': settings.startup.check_model,
            'ok': True,
            'detail': 'skipped',
            'path': str(model_path),
        },
        'database': {
            'enabled': settings.startup.check_database,
            'ok': True,
            'detail': 'skipped',
        },
    }

    if settings.startup.check_model:
        model_ok = model_path.exists() and model_path.is_file()
        checks['model']['ok'] = model_ok
        checks['model']['detail'] = 'ok' if model_ok else 'model file missing'

    if settings.startup.check_database:
        try:
            db.session.execute(text('SELECT 1'))
            checks['database']['detail'] = 'ok'
        except Exception as exc:
            checks['database']['ok'] = False
            checks['database']['detail'] = str(exc)

    healthy = checks['model']['ok'] and checks['database']['ok']
    return jsonify({
        'status': 'ok' if healthy else 'degraded',
        'checks': checks,
    }), (200 if healthy else 503)


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


@bp.route('/video/session/<int:detection_id>', methods=['POST'])
def create_video_session(detection_id):
    """创建视频处理会话"""
    detection = Detection.query.get_or_404(detection_id)
    if detection.file_type != 'video':
        return jsonify({"error": "Not a video file"}), 400

    video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], detection.filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    existing = session_manager.get_by_detection(detection_id)
    if existing and existing.state in (SessionState.RUNNING, SessionState.PAUSED):
        return jsonify(existing.to_dict())

    session = session_manager.create_session(detection_id, video_path)
    return jsonify(session.to_dict())


@bp.route('/video/session/<int:detection_id>/stats')
def get_video_stats(detection_id):
    """获取视频处理统计"""
    session = session_manager.get_by_detection(detection_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session.to_dict())


@bp.route('/video/session/<int:detection_id>/control', methods=['POST'])
def control_video_session(detection_id):
    """控制视频处理（暂停/继续/停止）"""
    session = session_manager.get_by_detection(detection_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    action = request.json.get('action') if request.is_json else request.form.get('action')
    if action == 'pause':
        session_manager.set_state(session.session_id, SessionState.PAUSED)
    elif action == 'resume':
        session_manager.set_state(session.session_id, SessionState.RUNNING)
    elif action == 'stop':
        session_manager.set_state(session.session_id, SessionState.STOPPED)
    else:
        return jsonify({"error": "Invalid action"}), 400

    return jsonify(session.to_dict())


@bp.route('/video/stream/<int:detection_id>')
def video_detection_stream(detection_id):
    """视频检测 MJPEG 流"""
    session = session_manager.get_by_detection(detection_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    def generate():
        cap = None
        frame_count = 0
        passenger_count = 0
        helmet_count = 0

        try:
            cap = cv2.VideoCapture(session.video_path)
            if not cap.isOpened():
                session_manager.set_state(session.session_id, SessionState.ERROR, "Cannot open video")
                return

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if not video_fps or video_fps <= 0:
                video_fps = 30.0

            session_manager.set_state(session.session_id, SessionState.RUNNING)
            session.stats.start_time = time.time()
            fps_start = time.time()
            fps_frames = 0

            while cap.isOpened():
                if session.state == SessionState.STOPPED:
                    break
                if session.state == SessionState.PAUSED:
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                detections = detector.detect(frame)
                violations, new_violations = detector.detect_violations(
                    detections, use_tracking=True
                )
                result_frame = detector.draw_results(frame, detections, violations)

                # Persist only newly emitted violation events to avoid per-frame duplicates.
                for v in new_violations:
                    violation_type = v.get('type')
                    if violation_type == 'passenger':
                        passenger_count += 1
                    elif violation_type == 'no_helmet':
                        helmet_count += 1

                    db.session.add(Violation(
                        detection_id=detection_id,
                        violation_type=violation_type,
                        frame_number=frame_count,
                        timestamp=frame_count / video_fps,
                        bbox=v.get('bbox'),
                        confidence=v.get('confidence', 0.0),
                    ))

                if new_violations:
                    db.session.commit()

                frame_count += 1
                fps_frames += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    current_fps = fps_frames / elapsed
                    session_manager.update_stats(
                        session.session_id,
                        processed_frames=frame_count,
                        fps=current_fps,
                        passenger_violations=passenger_count,
                        helmet_violations=helmet_count,
                    )
                    fps_start = time.time()
                    fps_frames = 0

                _, buffer = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except GeneratorExit:
            pass
        except Exception as exc:  # pragma: no cover - runtime dependent
            db.session.rollback()
            session_manager.set_state(session.session_id, SessionState.ERROR, str(exc))
        finally:
            if cap is not None:
                cap.release()
            session_manager.update_stats(
                session.session_id,
                processed_frames=frame_count,
                passenger_violations=passenger_count,
                helmet_violations=helmet_count,
            )
            if session.state not in (SessionState.STOPPED, SessionState.ERROR):
                session_manager.set_state(session.session_id, SessionState.COMPLETED)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
