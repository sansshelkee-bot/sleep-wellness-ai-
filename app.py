import os
from flask import Flask, render_template, session, redirect, url_for
from config import Config
from models.db_models import db, User
from routes import (
    auth_bp, dashboard_bp, prediction_bp,
    digital_twin_bp, simulator_bp, report_bp,
    facial_bp, consultation_bp
)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Database setup with graceful fallback if MySQL is offline
    try:
        db.init_app(app)
        with app.app_context():
            db.create_all()
            try:
                from sqlalchemy import inspect, text
                inspector = inspect(db.engine)
                if 'digital_twins' in inspector.get_table_names():
                    cols = [c['name'] for c in inspector.get_columns('digital_twins')]
                    with db.engine.connect() as conn:
                        if 'latest_facial_score' not in cols:
                            conn.execute(text("ALTER TABLE digital_twins ADD COLUMN latest_facial_score FLOAT NULL"))
                        if 'latest_facial_level' not in cols:
                            conn.execute(text("ALTER TABLE digital_twins ADD COLUMN latest_facial_level VARCHAR(20) NULL"))
                        if 'last_facial_assessment_date' not in cols:
                            conn.execute(text("ALTER TABLE digital_twins ADD COLUMN last_facial_assessment_date DATETIME NULL"))
                        conn.commit()
            except Exception:
                pass
    except Exception as e:
        print(f"[Database Warning] Could not connect to MySQL: {e}")
        # Fallback to local SQLite if MySQL is unavailable
        sqlite_fallback = f"sqlite:///{Config.BASE_DIR / 'sleep_wellness_fallback.db'}"
        print(f"[Database] Falling back to SQLite: {sqlite_fallback}")
        app.config['SQLALCHEMY_DATABASE_URI'] = sqlite_fallback
        db.init_app(app)
        with app.app_context():
            db.create_all()

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(digital_twin_bp)
    app.register_blueprint(simulator_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(facial_bp)
    app.register_blueprint(consultation_bp)

    # Landing page route
    @app.route('/')
    def index():
        if 'user_id' in session:
            return redirect(url_for('dashboard.dashboard_view'))
        return render_template('index.html')

    # Global context processor
    @app.context_processor
    def inject_global_vars():
        current_user = None
        if 'user_id' in session:
            current_user = db.session.get(User, session['user_id'])
        return {
            'current_user': current_user,
            'app_title': 'SleepTwin AI',
            'disclaimer_short': 'This system is intended for educational and wellness awareness purposes and does not replace professional medical diagnosis.'
        }

    # Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('base.html', content="<div class='container py-5 text-center'><h2>404 - Page Not Found</h2><p>The requested page does not exist.</p><a href='/' class='btn btn-primary'>Return Home</a></div>"), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('base.html', content="<div class='container py-5 text-center'><h2>500 - Internal Server Error</h2><p>An unexpected error occurred. Please try again.</p><a href='/' class='btn btn-primary'>Return Home</a></div>"), 500

    return app


app = create_app()

if __name__ == '__main__':
    # Running locally on Windows
    app.run(host='127.0.0.1', port=5000, debug=True)
