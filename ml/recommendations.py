def generate_personalized_recommendations(assessment_dict, prediction_result=None):
    """
    Generates tailored, actionable wellness recommendations across four pillars:
    1. Immediate Actions (Tonight / Next 24 hrs)
    2. Lifestyle Improvements (Daily routine & physical health)
    3. Sleep Habits (Circadian rhythm & bedtime environment)
    4. When to Seek Professional Help (Clinical red flags & evaluation)

    Recommendations adapt dynamically to the user's specific inputs and predicted disorder.
    """
    pred_class = prediction_result.get('prediction', 'None') if prediction_result else 'None'
    confidence = prediction_result.get('confidence', 0.0) if prediction_result else 0.0

    duration = float(assessment_dict.get('sleep_duration', 7.0))
    quality = int(assessment_dict.get('quality_of_sleep', 7))
    stress = int(assessment_dict.get('stress_level', 5))
    activity = float(assessment_dict.get('physical_activity_level', 45))
    steps = int(assessment_dict.get('daily_steps', 6000))
    bmi = str(assessment_dict.get('bmi_category', 'Normal')).strip().title()
    bp = str(assessment_dict.get('blood_pressure', '120/80'))
    hr = int(assessment_dict.get('heart_rate', 70))

    immediate_actions = []
    lifestyle_improvements = []
    sleep_habits = []
    seek_professional_help = []

    # 1. Immediate Actions
    if stress >= 6:
        immediate_actions.append({
            'text': "Practice 10 minutes of 4-7-8 deep breathing or progressive muscle relaxation before bed to downregulate stress arousal.",
            'priority': 'High'
        })
    else:
        immediate_actions.append({
            'text': "Begin winding down 45 minutes before sleep with dim ambient lighting and zero stimulating work tasks.",
            'priority': 'Medium'
        })

    if duration < 6.5:
        immediate_actions.append({
            'text': f"Target a bedtime 30 minutes earlier tonight to address your current {duration}-hour sleep deficit.",
            'priority': 'High'
        })
    else:
        immediate_actions.append({
            'text': "Ensure bedroom temperature is cool (ideally 18–20°C / 65–68°F) for thermoregulatory sleep onset.",
            'priority': 'Medium'
        })

    immediate_actions.append({
        'text': "Avoid caffeine, nicotine, and heavy meals at least 6 hours prior to your scheduled sleep time.",
        'priority': 'Medium'
    })

    # 2. Lifestyle Improvements
    if activity < 40:
        lifestyle_improvements.append({
            'text': f"Your recorded activity is {activity} mins/day. Aim for at least 30–45 minutes of moderate daytime exercise (brisk walking, cycling).",
            'priority': 'High'
        })
    else:
        lifestyle_improvements.append({
            'text': "Great daytime activity! Maintain regular physical training, but avoid vigorous workouts within 2 hours of bedtime.",
            'priority': 'Medium'
        })

    if steps < 6000:
        lifestyle_improvements.append({
            'text': f"Increase step count from {steps:,} towards 8,000 steps daily by integrating brief 5-minute movement breaks throughout work hours.",
            'priority': 'Medium'
        })

    if 'Overweight' in bmi or 'Obese' in bmi:
        lifestyle_improvements.append({
            'text': "Weight management and anti-inflammatory nutrition support upper airway stability and improve nocturnal respiration.",
            'priority': 'High'
        })

    if hr > 78:
        lifestyle_improvements.append({
            'text': f"Resting heart rate is {hr} bpm. Daytime hydration and cardiovascular conditioning will help lower resting autonomic tone.",
            'priority': 'Medium'
        })

    # 3. Sleep Habits & Hygiene
    sleep_habits.append({
        'text': "Anchor your circadian clock: wake up at the exact same hour every morning, even on weekends.",
        'priority': 'High'
    })

    if quality < 6:
        sleep_habits.append({
            'text': f"Your sleep quality rating is {quality}/10. Keep the bedroom completely dark using blackout curtains or a comfortable eye mask.",
            'priority': 'High'
        })
    else:
        sleep_habits.append({
            'text': "Expose your eyes to natural sunlight for 15 minutes within an hour of waking to boost daytime alertness and melatonin rhythm.",
            'priority': 'Medium'
        })

    sleep_habits.append({
        'text': "Reserve the bed strictly for sleep. If unable to fall asleep within 20 minutes, get up and read a book under soft light until drowsy.",
        'priority': 'Medium'
    })

    # 4. When to Seek Professional Help
    if pred_class == 'Sleep Apnea':
        seek_professional_help.append({
            'text': "The model detected high indicators consistent with Sleep Apnea. Consult an ENT or sleep specialist for an overnight sleep study (polysomnography).",
            'priority': 'High'
        })
        seek_professional_help.append({
            'text': "Seek immediate medical evaluation if you or your partner observe loud chronic snoring, gasping for air, or morning headaches.",
            'priority': 'High'
        })
    elif pred_class == 'Insomnia':
        seek_professional_help.append({
            'text': "The model detected high indicators of Insomnia. Consider consulting a licensed behavioral sleep specialist regarding CBT-I (Cognitive Behavioral Therapy for Insomnia).",
            'priority': 'High'
        })
        seek_professional_help.append({
            'text': "Seek medical advice if difficulty falling or staying asleep persists for more than 3 nights per week for over a month.",
            'priority': 'High'
        })
    else:
        seek_professional_help.append({
            'text': "Your current model assessment is Healthy (No disorder detected). Continue annual wellness checkups and routine blood pressure monitoring.",
            'priority': 'Low'
        })

    seek_professional_help.append({
        'text': "Always consult a certified healthcare practitioner before starting any sleep aids, supplements, or significant medical protocols.",
        'priority': 'Medium'
    })

    return {
        'immediate_actions': immediate_actions,
        'lifestyle_improvements': lifestyle_improvements,
        'sleep_habits': sleep_habits,
        'seek_professional_help': seek_professional_help,
        'disclaimer': 'These recommendations are wellness suggestions for educational purposes and do not constitute a medical prescription or clinical diagnosis.'
    }
