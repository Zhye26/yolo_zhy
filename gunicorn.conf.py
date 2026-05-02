"""Gunicorn configuration for production deployment."""
import os

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "2"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

preload_app = False
worker_tmp_dir = "/dev/shm"
