import re

def compute_sleep_health_score(assessment_dict, prediction_result=None):
    """
    Computes a deterministic, transparent Sleep Health Score (0–100)
    based on sleep architecture, lifestyle factors, and physiological metrics.

    Components:
    1. Sleep Duration Score (25%)
    2. Sleep Quality Score (25%)
    3. Stress Management Score (20%)
    4. Physical Activity & Steps Score (15%)
    5. Cardiovascular & BMI Wellness Score (15%)
    With minor adjustment based on predicted disorder probability.

    Classification:
      70 - 100: Good
      40 - 69:  Moderate
      0  - 39:  Needs Attention
    """
    # 1. Sleep Duration Component (25%)
    duration = float(assessment_dict.get('sleep_duration', 7.0))
    if 7.0 <= duration <= 8.5:
        duration_score = 100.0
    elif 6.5 <= duration < 7.0:
        duration_score = 88.0
    elif 6.0 <= duration < 6.5:
        duration_score = 75.0
    elif 5.5 <= duration < 6.0:
        duration_score = 60.0
    elif 5.0 <= duration < 5.5:
        duration_score = 45.0
    elif duration > 8.5 and duration <= 9.5:
        duration_score = 85.0
    elif duration > 9.5:
        duration_score = 65.0
    else:  # < 5.0 hours
        duration_score = max(20.0, duration * 6)

    # 2. Sleep Quality Component (25%)
    quality = int(assessment_dict.get('quality_of_sleep', 7))
    quality = max(1, min(10, quality))
    quality_score = float(quality * 10)

    # 3. Stress Management Component (20%)
    stress = int(assessment_dict.get('stress_level', 5))
    stress = max(1, min(10, stress))
    stress_score = float((10 - stress + 1) * 10)
    stress_score = max(10.0, min(100.0, stress_score))

    # 4. Physical Activity & Steps Component (15%)
    activity_mins = float(assessment_dict.get('physical_activity_level', 45))
    steps = int(assessment_dict.get('daily_steps', 6000))

    # Activity score: 60 mins is 100%
    act_sub = min(100.0, (activity_mins / 60.0) * 100.0)
    # Steps score: 8,000 steps is 100%
    step_sub = min(100.0, (steps / 8000.0) * 100.0)
    activity_score = round(0.5 * act_sub + 0.5 * step_sub, 1)

    # 5. Cardiovascular & BMI Component (15%)
    hr = int(assessment_dict.get('heart_rate', 70))
    bmi = str(assessment_dict.get('bmi_category', 'Normal')).strip().title()

    # BMI factor
    if 'Normal' in bmi:
        bmi_sub = 100.0
    elif 'Overweight' in bmi:
        bmi_sub = 70.0
    else:  # Obese
        bmi_sub = 45.0

    # Heart Rate factor (ideal resting HR 55-72)
    if 55 <= hr <= 72:
        hr_sub = 100.0
    elif 73 <= hr <= 80:
        hr_sub = 85.0
    elif 81 <= hr <= 90:
        hr_sub = 65.0
    else:
        hr_sub = 45.0

    cardio_score = round(0.5 * bmi_sub + 0.5 * hr_sub, 1)

    # Base Weighted Total
    weighted_score = (
        0.25 * duration_score +
        0.25 * quality_score +
        0.20 * stress_score +
        0.15 * activity_score +
        0.15 * cardio_score
    )

    # Risk Penalty adjustment based on model prediction
    penalty = 0.0
    if prediction_result:
        pred_label = prediction_result.get('prediction', 'None')
        probs = prediction_result.get('probabilities', {})
        if pred_label == 'Sleep Apnea':
            prob_val = probs.get('Sleep Apnea', 0.8)
            penalty = 12.0 * prob_val
        elif pred_label == 'Insomnia':
            prob_val = probs.get('Insomnia', 0.8)
            penalty = 10.0 * prob_val

    final_score = max(5.0, min(100.0, weighted_score - penalty))
    final_score = round(final_score, 1)

    # Classification Category
    if final_score >= 70.0:
        category = "Good"
        color_class = "success"
        badge_text = "Optimal Sleep Health"
    elif final_score >= 40.0:
        category = "Moderate"
        color_class = "warning"
        badge_text = "Moderate Wellness"
    else:
        category = "Needs Attention"
        color_class = "danger"
        badge_text = "Attention Recommended"

    return {
        'overall_score': final_score,
        'category': category,
        'color_class': color_class,
        'badge_text': badge_text,
        'breakdown': {
            'sleep_duration': round(duration_score, 1),
            'sleep_quality': round(quality_score, 1),
            'stress_management': round(stress_score, 1),
            'physical_activity': round(activity_score, 1),
            'cardiovascular_wellness': round(cardio_score, 1)
        },
        'disclaimer': 'This score is a deterministic wellness indicator for educational guidance and does not replace medical diagnostics.'
    }
