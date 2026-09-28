from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from models.db_models import db, SleepAssessment, Prediction, ShapExplanation, FutureRisk, Recommendation, DigitalTwin, FacialAssessment
from services.digital_twin_service import DigitalTwinService
from utils.helpers import login_required, get_current_user
from ml.predict import predict_sleep_disorder
from ml.explain import explain_prediction
from ml.scoring import compute_sleep_health_score
from ml.risk import calculate_future_risk
from ml.recommendations import generate_personalized_recommendations

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/assessment', methods=['GET'])
@login_required
def assessment_form():
    user = get_current_user()
    latest_assessment = SleepAssessment.query.filter_by(user_id=user.id).order_by(SleepAssessment.created_at.desc()).first()
    latest_facial = FacialAssessment.query.filter_by(user_id=user.id).order_by(FacialAssessment.assessment_date.desc()).first()
    recent_facial_assessments = FacialAssessment.query.filter_by(user_id=user.id).order_by(FacialAssessment.assessment_date.desc()).limit(10).all()
    twin_data = DigitalTwinService.get_digital_twin_state(user.id)
    active_tab = request.args.get('tab', 'clinical')
    return render_template(
        'assessment.html',
        user=user,
        latest=latest_assessment,
        latest_facial=latest_facial,
        recent_assessments=[a.to_dict() for a in recent_facial_assessments],
        twin_data=twin_data,
        active_tab=active_tab
    )


@prediction_bp.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """
    Primary API endpoint for submitting sleep assessments and running ML prediction.
    Accepts JSON or Form data.
    """
    user_id = session.get('user_id')
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    try:
        result = DigitalTwinService.process_new_assessment(user_id, data)
        return jsonify({
            'status': 'success',
            'assessment_id': result['assessment_id'],
            'prediction': result['prediction']['prediction'],
            'confidence': result['prediction']['confidence'],
            'probabilities': result['prediction']['probabilities'],
            'health_score': result['health_score']['overall_score'],
            'health_score_category': result['health_score']['category'],
            'future_risk': result['future_risk']['risk_level'],
            'risk_score': result['future_risk']['risk_score'],
            'digital_twin': result['digital_twin'],
            'redirect_url': url_for('prediction.prediction_result_view', assessment_id=result['assessment_id'])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@prediction_bp.route('/prediction/<int:assessment_id>', methods=['GET'])
@login_required
def prediction_result_view(assessment_id):
    user_id = session.get('user_id')
    assessment = SleepAssessment.query.filter_by(id=assessment_id, user_id=user_id).first_or_404()
    prediction = Prediction.query.filter_by(assessment_id=assessment.id).first_or_404()
    shap_factors = ShapExplanation.query.filter_by(assessment_id=assessment.id).order_by(ShapExplanation.shap_value.desc()).all()
    future_risk = FutureRisk.query.filter_by(assessment_id=assessment.id).first()
    recommendations = Recommendation.query.filter_by(assessment_id=assessment.id).all()
    twin = DigitalTwin.query.filter_by(user_id=user_id).first()

    # Group recommendations by pillar
    grouped_recs = {
        'Immediate Actions': [r for r in recommendations if r.category == 'Immediate Actions'],
        'Lifestyle Improvements': [r for r in recommendations if r.category == 'Lifestyle Improvements'],
        'Sleep Habits': [r for r in recommendations if r.category == 'Sleep Habits'],
        'When to Seek Professional Help': [r for r in recommendations if r.category == 'When to Seek Professional Help']
    }

    # Reconstruct score result for detailed display
    score_details = compute_sleep_health_score(assessment.to_dict(), prediction.to_dict())

    return render_template(
        'prediction.html',
        assessment=assessment,
        prediction=prediction,
        shap_factors=shap_factors,
        future_risk=future_risk,
        grouped_recs=grouped_recs,
        score=score_details,
        twin=twin
    )


@prediction_bp.route('/api/explain', methods=['POST'])
@login_required
def api_explain():
    """
    On-demand SHAP explanation calculation for provided features.
    """
    data = request.get_json() if request.is_json else request.form.to_dict()
    try:
        shap_result = explain_prediction(data)
        return jsonify(shap_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@prediction_bp.route('/api/score', methods=['POST'])
@login_required
def api_score():
    data = request.get_json() if request.is_json else request.form.to_dict()
    try:
        score_result = compute_sleep_health_score(data)
        return jsonify(score_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@prediction_bp.route('/api/future-risk', methods=['POST'])
@login_required
def api_future_risk():
    data = request.get_json() if request.is_json else request.form.to_dict()
    try:
        risk_result = calculate_future_risk(data)
        return jsonify(risk_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@prediction_bp.route('/api/recommendations', methods=['POST'])
@login_required
def api_recommendations():
    data = request.get_json() if request.is_json else request.form.to_dict()
    try:
        recs = generate_personalized_recommendations(data)
        return jsonify(recs)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
