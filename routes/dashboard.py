import json
from flask import Blueprint, render_template, jsonify, session
from models.db_models import SleepAssessment, Prediction, DigitalTwin, ProgressRecord, FacialAssessment
from services.digital_twin_service import DigitalTwinService
from utils.helpers import login_required, get_current_user
from config import Config

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@login_required
def dashboard_view():
    user = get_current_user()
    twin_data = DigitalTwinService.get_digital_twin_state(user.id)
    latest_assessment = SleepAssessment.query.filter_by(user_id=user.id).order_by(SleepAssessment.created_at.desc()).first()
    latest_facial = FacialAssessment.query.filter_by(user_id=user.id).order_by(FacialAssessment.assessment_date.desc()).first()
    
    total_assessments = SleepAssessment.query.filter_by(user_id=user.id).count()
    all_assessments = SleepAssessment.query.filter_by(user_id=user.id).order_by(SleepAssessment.created_at.desc()).limit(10).all()

    return render_template(
        'dashboard.html',
        user=user,
        twin_data=twin_data,
        latest=latest_assessment,
        latest_facial=latest_facial,
        total_assessments=total_assessments,
        recent_assessments=all_assessments
    )


@dashboard_bp.route('/history')
@login_required
def history_view():
    user = get_current_user()
    assessments = SleepAssessment.query.filter_by(user_id=user.id).order_by(SleepAssessment.created_at.desc()).all()
    facial_assessments = FacialAssessment.query.filter_by(user_id=user.id).order_by(FacialAssessment.assessment_date.desc()).all()
    return render_template('history.html', user=user, assessments=assessments, facial_assessments=facial_assessments)


@dashboard_bp.route('/model-info')
def model_info_view():
    """
    Viva / Academic Model Details page displaying real evaluated metrics,
    CatBoost, Random Forest, Gradient Boosting, Voting Ensemble comparisons.
    """
    user = get_current_user()
    metrics_data = {}
    metadata_data = {}

    if Config.METRICS_PATH.exists():
        try:
            with open(Config.METRICS_PATH, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")

    if Config.METADATA_PATH.exists():
        try:
            with open(Config.METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")

    return render_template(
        'model_info.html',
        user=user,
        metrics=metrics_data,
        metadata=metadata_data
    )


@dashboard_bp.route('/api/progress')
@login_required
def api_progress():
    user_id = session.get('user_id')
    data = DigitalTwinService.get_progress_data(user_id)
    return jsonify(data)


@dashboard_bp.route('/api/history')
@login_required
def api_history():
    user_id = session.get('user_id')
    assessments = SleepAssessment.query.filter_by(user_id=user_id).order_by(SleepAssessment.created_at.desc()).all()
    results = []
    for a in assessments:
        item = a.to_dict()
        if a.prediction:
            item['prediction'] = a.prediction.predicted_disorder
            item['confidence'] = round(a.prediction.confidence * 100, 1)
        if a.progress_record:
            item['health_score'] = a.progress_record.health_score
            item['risk_level'] = a.progress_record.risk_level
        results.append(item)
    return jsonify(results)
