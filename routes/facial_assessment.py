import base64
from flask import Blueprint, render_template, request, jsonify, session
from utils.helpers import login_required
from models.db_models import db, User, DigitalTwin, FacialAssessment
from services.digital_twin_service import DigitalTwinService
from ml.facial_analyzer import compute_facial_sleepiness_score, FrameQualityValidator

facial_bp = Blueprint('facial', __name__)
validator = FrameQualityValidator()


@facial_bp.route('/facial-assessment', methods=['GET'])
@login_required
def facial_assessment_view():
    """
    Redirects to the unified Multimodal Health & Facial Assessment page.
    """
    from flask import redirect, url_for
    return redirect(url_for('prediction.assessment_form', tab='facial'))


@facial_bp.route('/api/facial-assessment/submit', methods=['POST'])
@login_required
def submit_facial_assessment():
    """
    Receives quantitative biometric session telemetry from the browser,
    runs the explainable scoring engine, and updates the user's Digital Twin.
    """
    user_id = session.get('user_id')
    data = request.get_json() or {}

    try:
        # Run scoring & explainability engine
        assessment_result = compute_facial_sleepiness_score(data)

        # Persist and synchronize with Digital Twin
        saved_entity = DigitalTwinService.record_facial_assessment(user_id, assessment_result)

        # Retrieve updated digital twin state
        twin_data = DigitalTwinService.get_digital_twin_state(user_id)

        return jsonify({
            'status': 'success',
            'assessment_id': saved_entity.id,
            'result': saved_entity.to_dict(),
            'digital_twin': twin_data['twin'] if twin_data else None,
            'message': 'Facial sleepiness assessment synchronized successfully with your Digital Twin.'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Assessment processing failed: {str(e)}'
        }), 400


@facial_bp.route('/api/facial-assessment/analyze-frame', methods=['POST'])
@login_required
def analyze_frame():
    """
    Backend validation endpoint: analyzes an individual webcam frame for face tracking,
    lighting quality, and camera alignment.
    """
    data = request.get_json() or {}
    image_b64 = data.get('image', '')

    if not image_b64:
        return jsonify({'status': 'warning', 'code': 'NO_IMAGE', 'message': 'No image data received.'}), 400

    try:
        # Strip data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',', 1)[1]

        image_bytes = base64.b64decode(image_b64)
        validation_res = validator.analyze_frame_bytes(image_bytes)
        return jsonify(validation_res)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@facial_bp.route('/api/facial-assessment/history', methods=['GET'])
@login_required
def facial_history_api():
    """
    Returns time-series telemetry data for the user's facial assessments.
    """
    user_id = session.get('user_id')
    history_data = DigitalTwinService.get_facial_progress_data(user_id)
    return jsonify(history_data)
