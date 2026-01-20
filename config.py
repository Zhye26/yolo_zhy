import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'mysql+pymysql://root:password@localhost/ebike_detection'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
    OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/outputs')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

    # YOLO settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/best.pt')
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45

    # Detection classes
    CLASS_NAMES = ['ebike', 'driver', 'passenger', 'helmet']
