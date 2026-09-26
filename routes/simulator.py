from flask import Blueprint, render_template, request, jsonify, session
from models.db_models import DigitalTwin, SleepAssessment, Prediction
from services.digital_twin_service import DigitalTwinService
from utils.helpers import login_required, get_current_user
from ml.predict import predict_sleep_disorder
from ml.scoring import compute_sleep_health_score
from ml.risk import calculate_future_risk
from ml.recommendations import generate_personalized_recommendations

simulator_bp = Blueprint('simulator', __name__)

@simulator_bp.route('/simulator')
@login_required
def simulator_view():
    user = get_current_user()
    twin = DigitalTwin.query.filter_by(user_id=user.id).first()
    latest = SleepAssessment.query.filter_by(user_id=user.id).order_by(SleepAssessment.created_at.desc()).first()

    # Default baseline parameters if no assessment yet
    baseline = {
        'age': latest.age if latest else (user.age or 35),
        'gender': latest.gender if latest else (user.gender or 'Male'),
        'occupation': latest.occupation if latest else (user.occupation or 'Engineer'),
        'sleep_duration': latest.sleep_duration if latest else 6.0,
        'quality_of_sleep': latest.quality_of_sleep if latest else 6,
        'physical_activity_level': latest.physical_activity_level if latest else 35.0,
        'stress_level': latest.stress_level if latest else 7,
        'bmi_category': latest.bmi_category if latest else 'Overweight',
        'blood_pressure': latest.blood_pressure if latest else '130/85',
        'heart_rate': latest.heart_rate if latest else 78,
        'daily_steps': latest.daily_steps if latest else 5000,
        'predicted_disorder': twin.current_predicted_disorder if twin else 'Pending',
        'health_score': twin.current_health_score if twin else 60.0,
        'future_risk': twin.current_future_risk if twin else 'Moderate'
    }

    return render_template('simulator.html', user=user, baseline=baseline)


@simulator_bp.route('/api/simulate', methods=['POST'])
@login_required
def api_simulate():
    """
    Simulates lifestyle and health modifications without persisting changes to the database.
    Returns side-by-side comparison of baseline vs simulated metrics.
    """
    user_id = session.get('user_id')
    user = get_current_user()
    sim_data = request.get_json() if request.is_json else request.form.to_dict()

    try:
        # Base input format
        sim_input = {
            'Age': float(sim_data.get('age', user.age or 35)),
            'Gender': str(sim_data.get('gender', user.gender or 'Male')),
            'Occupation': str(sim_data.get('occupation', user.occupation or 'Engineer')),
            'Sleep Duration': float(sim_data.get('sleep_duration', 7.5)),
            'Quality of Sleep': int(sim_data.get('quality_of_sleep', 8)),
            'Physical Activity Level': float(sim_data.get('physical_activity_level', 60.0)),
            'Stress Level': int(sim_data.get('stress_level', 4)),
            'BMI Category': str(sim_data.get('bmi_category', 'Normal')),
            'Blood Pressure': str(sim_data.get('blood_pressure', '120/80')),
            'Heart Rate': int(sim_data.get('heart_rate', 70)),
            'Daily Steps': int(sim_data.get('daily_steps', 8000))
        }

        # 1. Run simulation prediction
        pred_res = predict_sleep_disorder(sim_input)

        # 2. Run simulation score
        score_res = compute_sleep_health_score(sim_input, pred_res)

        # 3. Run simulation risk
        risk_res = calculate_future_risk(sim_input, pred_res)

        # 4. Run simulation recommendations
        rec_res = generate_personalized_recommendations(sim_input, pred_res)

        # Fetch current saved twin for comparison
        twin = DigitalTwin.query.filter_by(user_id=user_id).first()
        baseline_score = twin.current_health_score if twin and twin.current_health_score else 60.0
        score_diff = round(score_res['overall_score'] - baseline_score, 1)

        return jsonify({
            'status': 'success',
            'simulated_prediction': pred_res['prediction'],
            'simulated_confidence': pred_res['confidence'],
            'simulated_probabilities': pred_res['probabilities'],
            'simulated_score': score_res,
            'simulated_risk': risk_res,
            'simulated_recommendations': rec_res,
            'score_diff': score_diff,
            'comparison': {
                'baseline_disorder': twin.current_predicted_disorder if twin else 'Unassessed',
                'baseline_score': baseline_score,
                'baseline_risk': twin.current_future_risk if twin else 'Moderate',
                'simulated_disorder': pred_res['prediction'],
                'simulated_score': score_res['overall_score'],
                'simulated_risk': risk_res['risk_level']
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
