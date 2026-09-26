from flask import Blueprint, render_template, jsonify, session
from models.db_models import DigitalTwin, SleepAssessment, Prediction, ShapExplanation, FutureRisk
from services.digital_twin_service import DigitalTwinService
from utils.helpers import login_required, get_current_user

digital_twin_bp = Blueprint('digital_twin', __name__)

@digital_twin_bp.route('/digital-twin')
@login_required
def digital_twin_view():
    user = get_current_user()
    twin_state = DigitalTwinService.get_digital_twin_state(user.id)
    
    # Get latest assessment SHAP factors and future risk
    latest_assessment = SleepAssessment.query.filter_by(user_id=user.id).order_by(SleepAssessment.created_at.desc()).first()
    shap_factors = []
    future_risk = None
    if latest_assessment:
        shap_factors = ShapExplanation.query.filter_by(assessment_id=latest_assessment.id).order_by(ShapExplanation.shap_value.desc()).limit(5).all()
        future_risk = FutureRisk.query.filter_by(assessment_id=latest_assessment.id).first()

    return render_template(
        'digital_twin.html',
        user=user,
        twin_data=twin_state,
        latest=latest_assessment,
        shap_factors=shap_factors,
        future_risk=future_risk
    )


@digital_twin_bp.route('/api/digital-twin')
@login_required
def api_digital_twin():
    user_id = session.get('user_id')
    twin_state = DigitalTwinService.get_digital_twin_state(user_id)
    return jsonify(twin_state)
