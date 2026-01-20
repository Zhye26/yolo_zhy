import os
import cv2
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename
from app.services import EbikeDetector, VideoProcessor
from app.models import Detection, Violation
from app import db

bp = Blueprint('detection', __name__)

detector = EbikeDetector()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    """上传文件页面"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('没有选择文件')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('没有选择文件')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 判断文件类型
            ext = filename.rsplit('.', 1)[1].lower()
            file_type = 'video' if ext in {'mp4', 'avi', 'mov'} else 'image'

            # 创建检测记录
            detection = Detection(filename=filename, file_type=file_type)
            db.session.add(detection)
            db.session.commit()

            return redirect(url_for('detection.process', detection_id=detection.id))

    return render_template('upload.html')


@bp.route('/process/<int:detection_id>')
def process(detection_id):
    """处理检测"""
    detection = Detection.query.get_or_404(detection_id)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], detection.filename)

    if detection.file_type == 'image':
        # 图片检测
        image = cv2.imread(filepath)
        results = detector.detect(image)
        violations = detector.detect_violations(results)

        # 保存结果图片
        result_image = detector.draw_results(image, results, violations)
        output_filename = f"result_{detection.filename}"
        output_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_path, result_image)

        # 保存违规记录
        for v in violations:
            violation = Violation(
                detection_id=detection.id,
                violation_type=v['type'],
                bbox=v['bbox'],
                confidence=v['confidence']
            )
            db.session.add(violation)

        db.session.commit()

        return render_template('result.html',
                             detection=detection,
                             violations=violations,
                             output_filename=output_filename)
    else:
        # 视频检测 - 跳转到视频处理页面
        return render_template('video_process.html', detection=detection)


@bp.route('/result/<int:detection_id>')
def result(detection_id):
    """查看检测结果"""
    detection = Detection.query.get_or_404(detection_id)
    violations = detection.violations.all()
    output_filename = f"result_{detection.filename}"

    return render_template('result.html',
                         detection=detection,
                         violations=violations,
                         output_filename=output_filename)
