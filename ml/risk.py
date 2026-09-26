def calculate_future_risk(assessment_dict, prediction_result=None, previous_assessments=None):
    """
    Calculates estimated future wellness risk index (0–100) and risk category
    (Low, Moderate, High), identifying contributing risk drivers and trend trajectory.

    Formula weights:
    - ML Disorder Probability (35%): Apnea / Insomnia probabilities
    - Chronic Sleep Deficit (20%): Sleep duration < 6 hrs or Quality < 5
    - Autonomic Stress Load (15%): Stress Level >= 7
    - Sedentary Lifestyle (15%): Activity < 30 mins, Steps < 5000
    - Metabolic & Cardiovascular Strain (15%): BMI Overweight/Obese, BP > 130/85, HR > 78
    """
    risk_points = 0.0
    risk_factors = []
    recommended_actions = []

    # 1. ML Disorder Probability (35 points max)
    if prediction_result:
        probs = prediction_result.get('probabilities', {})
        p_apnea = probs.get('Sleep Apnea', 0.0)
        p_insomnia = probs.get('Insomnia', 0.0)
        disorder_prob = max(p_apnea, p_insomnia)
        
        ml_risk_contribution = disorder_prob * 35.0
        risk_points += ml_risk_contribution
        
        if p_apnea > 0.4:
            risk_factors.append(f"High model probability ({round(p_apnea * 100)}%) for Sleep Apnea")
            recommended_actions.append("Consider clinical sleep screening (polysomnography) if snoring or breathing pauses occur.")
        elif p_insomnia > 0.4:
            risk_factors.append(f"Elevated model probability ({round(p_insomnia * 100)}%) for Chronic Insomnia")
            recommended_actions.append("Adopt sleep hygiene guidelines and cognitive relaxation techniques.")

    # 2. Chronic Sleep Deficit (20 points max)
    duration = float(assessment_dict.get('sleep_duration', 7.0))
    quality = int(assessment_dict.get('quality_of_sleep', 7))

    if duration < 5.5:
        risk_points += 15.0
        risk_factors.append(f"Severe sleep deficit ({duration} hrs/night; recommended 7-9 hrs)")
        recommended_actions.append("Gradually shift bedtime earlier by 15-30 minutes each week.")
    elif duration < 6.5:
        risk_points += 8.0
        risk_factors.append(f"Sub-optimal sleep duration ({duration} hrs/night)")

    if quality <= 4:
        risk_points += 8.0
        risk_factors.append(f"Low restorative sleep quality rating ({quality}/10)")
        recommended_actions.append("Minimize blue light screen exposure 60 minutes before sleeping.")
    elif quality <= 6:
        risk_points += 4.0

    # 3. Autonomic Stress Load (15 points max)
    stress = int(assessment_dict.get('stress_level', 5))
    if stress >= 8:
        risk_points += 15.0
        risk_factors.append(f"Elevated chronic stress index ({stress}/10)")
        recommended_actions.append("Incorporate parasympathetic relaxation like 4-7-8 breathing or mindfulness.")
    elif stress >= 6:
        risk_points += 8.0
        risk_factors.append(f"Moderate daily stress levels ({stress}/10)")

    # 4. Sedentary Lifestyle (15 points max)
    activity = float(assessment_dict.get('physical_activity_level', 40))
    steps = int(assessment_dict.get('daily_steps', 6000))

    if activity < 30:
        risk_points += 8.0
        risk_factors.append(f"Low daily physical activity ({activity} mins/day)")
        recommended_actions.append("Engage in at least 30 minutes of moderate aerobic activity most days.")
    if steps < 5000:
        risk_points += 7.0
        risk_factors.append(f"Sedentary step count ({steps:,} steps/day)")
        recommended_actions.append("Aim for regular walking breaks throughout workday.")

    # 5. Metabolic & Cardiovascular Strain (15 points max)
    bmi = str(assessment_dict.get('bmi_category', 'Normal')).strip().title()
    hr = int(assessment_dict.get('heart_rate', 70))
    bp = str(assessment_dict.get('blood_pressure', '120/80'))

    if 'Obese' in bmi:
        risk_points += 8.0
        risk_factors.append("Obesity category increases respiratory resistance during sleep")
        recommended_actions.append("Discuss personalized dietary and metabolic management with a clinician.")
    elif 'Overweight' in bmi:
        risk_points += 4.0
        risk_factors.append("Overweight BMI classification is a recognized risk contributor")

    if hr > 80:
        risk_points += 5.0
        risk_factors.append(f"Elevated resting heart rate ({hr} bpm)")

    # Parse Systolic BP if > 130
    try:
        systolic = int(bp.split('/')[0])
        if systolic >= 135:
            risk_points += 5.0
            risk_factors.append(f"Elevated systolic blood pressure ({systolic} mmHg)")
    except Exception:
        pass

    final_risk_score = round(min(100.0, max(5.0, risk_points)), 1)

    # Risk level categorization
    if final_risk_score >= 60.0:
        risk_level = "High"
        color_class = "danger"
    elif final_risk_score >= 30.0:
        risk_level = "Moderate"
        color_class = "warning"
    else:
        risk_level = "Low"
        color_class = "success"

    # Evaluate trajectory/trend if historical records are provided
    trajectory = "Stable"
    if previous_assessments and len(previous_assessments) > 0:
        prev_scores = [a.get('health_score', 70.0) for a in previous_assessments if 'health_score' in a]
        if prev_scores:
            avg_past = sum(prev_scores) / len(prev_scores)
            current_health = 100.0 - final_risk_score
            if current_health > avg_past + 5:
                trajectory = "Improving"
            elif current_health < avg_past - 5:
                trajectory = "Worsening"

    if not risk_factors:
        risk_factors = ["No adverse risk markers detected in current assessment profile."]
    if not recommended_actions:
        recommended_actions = ["Maintain current positive sleep routine and consistent physical activity."]

    preventive_summary = (
        "Focus on improving sleep duration consistency, managing autonomic stress before bedtime, "
        "and sustaining daily aerobic activity to support restorative sleep cycles."
    )

    return {
        'risk_level': risk_level,
        'risk_score': final_risk_score,
        'risk_trajectory': trajectory,
        'color_class': color_class,
        'primary_risk_factors': risk_factors[:5],
        'recommended_actions': recommended_actions[:4],
        'preventive_guidance': preventive_summary,
        'disclaimer': 'This is an estimated wellness risk indicator based on current available data, not a medical forecast.'
    }
