from pathlib import Path

from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

from app.config import Config, settings

db = SQLAlchemy()
migrate = Migrate()


def _run_startup_checks(app: Flask) -> None:
    """Run lightweight startup checks and optionally fail fast."""
    issues = []

    # Ensure configured storage paths are present.
    settings.ensure_dirs()

    if settings.startup.check_model:
        model_path = Path(settings.model.model_path)
        if not model_path.exists():
            issues.append(f"Model file not found: {model_path}")
        elif not model_path.is_file():
            issues.append(f"Model path is not a file: {model_path}")

    if settings.startup.check_database:
        try:
            with app.app_context():
                db.session.execute(text("SELECT 1"))
        except Exception as exc:  # pragma: no cover - depends on runtime env
            issues.append(f"Database connectivity check failed: {exc}")

    if issues:
        for issue in issues:
            app.logger.warning("[startup-check] %s", issue)
        if settings.startup.strict:
            raise RuntimeError("Startup checks failed: " + " | ".join(issues))
    else:
        app.logger.info("[startup-check] all checks passed")


def create_app(config_class=Config):
    app = Flask(
        __name__,
        template_folder='../templates',
        static_folder='../static',
    )
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)

    from app.routes import main, detection, api
    app.register_blueprint(main.bp)
    app.register_blueprint(detection.bp)
    app.register_blueprint(api.bp, url_prefix='/api')

    _run_startup_checks(app)

    return app
