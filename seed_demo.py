import os
from datetime import datetime, timezone, timedelta
from app import create_app
from models.db_models import (
    db, User, SleepAssessment, Prediction, DigitalTwin,
    Recommendation, ShapExplanation, FutureRisk, ProgressRecord
)
from ml.predict import predict_sleep_disorder
from ml.explain import explain_prediction
from ml.scoring import compute_sleep_health_score
from ml.risk import calculate_future_risk
from ml.recommendations import generate_personalized_recommendations

def seed_demo_data():
    app = create_app()
    with app.app_context():
        print("=" * 60)
        print("  SEEDING DEMONSTRATION USER & PROGRESSION RECORDS")
        print("=" * 60)

        demo_email = "demo@sleepwellness.ai"
        existing = User.query.filter_by(email=demo_email).first()
        if existing:
            print(f"[INFO] Demo user '{demo_email}' already exists. Refreshing records...")
            db.session.delete(existing)
            db.session.commit()

        # Create Demo User
        demo_user = User(
            full_name="Alex Taylor (DEMO)",
            email=demo_email,
            age=36,
            gender="Male",
            occupation="Software Engineer"
        )
        demo_user.set_password("password123")
        db.session.add(demo_user)
        db.session.commit()
        print(f"Created Demo User: {demo_user.email} (Password: password123)")

        # Create historical progression over 5 weeks
        # Week 1: High stress, short sleep -> Insomnia
        # Week 3: Moderate stress, improving routine -> Borderline
        # Week 5: Balanced sleep, high activity -> None (Healthy)
        sample_timeline = [
            {
                'days_ago': 28,
                'sleep_duration': 5.2,
                'quality_of_sleep': 4,
                'physical_activity_level': 25.0,
                'stress_level': 8,
                'bmi_category': 'Overweight',
                'blood_pressure': '134/86',
                'systolic_bp': 134,
                'diastolic_bp': 86,
                'heart_rate': 82,
                'daily_steps': 4200
            },
            {
                'days_ago': 21,
                'sleep_duration': 5.8,
                'quality_of_sleep': 5,
                'physical_activity_level': 35.0,
                'stress_level': 7,
                'bmi_category': 'Overweight',
                'blood_pressure': '130/84',
                'systolic_bp': 130,
                'diastolic_bp': 84,
                'heart_rate': 78,
                'daily_steps': 5500
            },
            {
                'days_ago': 14,
                'sleep_duration': 6.5,
                'quality_of_sleep': 6,
                'physical_activity_level': 45.0,
                'stress_level': 6,
                'bmi_category': 'Normal',
                'blood_pressure': '126/82',
                'systolic_bp': 126,
                'diastolic_bp': 82,
                'heart_rate': 74,
                'daily_steps': 6800
            },
            {
                'days_ago': 7,
                'sleep_duration': 7.1,
                'quality_of_sleep': 8,
                'physical_activity_level': 55.0,
                'stress_level': 4,
                'bmi_category': 'Normal',
                'blood_pressure': '122/80',
                'systolic_bp': 122,
                'diastolic_bp': 80,
                'heart_rate': 69,
                'daily_steps': 7800
            },
            {
                'days_ago': 1,
                'sleep_duration': 7.8,
                'quality_of_sleep': 9,
                'physical_activity_level': 65.0,
                'stress_level': 3,
                'bmi_category': 'Normal',
                'blood_pressure': '118/78',
                'systolic_bp': 118,
                'diastolic_bp': 78,
                'heart_rate': 66,
                'daily_steps': 9100
            }
        ]

        latest_assessment = None
        for step in sample_timeline:
            rec_date = datetime.now(timezone.utc) - timedelta(days=step['days_ago'])
            assessment = SleepAssessment(
                user_id=demo_user.id,
                age=demo_user.age,
                gender=demo_user.gender,
                occupation=demo_user.occupation,
                sleep_duration=step['sleep_duration'],
                quality_of_sleep=step['quality_of_sleep'],
                physical_activity_level=step['physical_activity_level'],
                stress_level=step['stress_level'],
                bmi_category=step['bmi_category'],
                blood_pressure=step['blood_pressure'],
                systolic_bp=step['systolic_bp'],
                diastolic_bp=step['diastolic_bp'],
                heart_rate=step['heart_rate'],
                daily_steps=step['daily_steps'],
                is_demo=True,
                created_at=rec_date
            )
            db.session.add(assessment)
            db.session.flush()

            features_dict = {
                'Age': assessment.age,
                'Gender': assessment.gender,
                'Occupation': assessment.occupation,
                'Sleep Duration': assessment.sleep_duration,
                'Quality of Sleep': assessment.quality_of_sleep,
                'Physical Activity Level': assessment.physical_activity_level,
                'Stress Level': assessment.stress_level,
                'BMI Category': assessment.bmi_category,
                'Blood Pressure': assessment.blood_pressure,
                'Heart Rate': assessment.heart_rate,
                'Daily Steps': assessment.daily_steps
            }

            pred_res = predict_sleep_disorder(features_dict)
            prediction = Prediction(
                assessment_id=assessment.id,
                user_id=demo_user.id,
                predicted_disorder=pred_res['prediction'],
                confidence=pred_res['confidence'],
                prob_none=pred_res['probabilities'].get('None', 0.0),
                prob_insomnia=pred_res['probabilities'].get('Insomnia', 0.0),
                prob_apnea=pred_res['probabilities'].get('Sleep Apnea', 0.0),
                model_version='CatBoost v1.2',
                created_at=rec_date
            )
            db.session.add(prediction)

            score_res = compute_sleep_health_score(features_dict, pred_res)
            shap_res = explain_prediction(features_dict, pred_res)
            for factor in shap_res.get('top_factors', [])[:5]:
                shap_entity = ShapExplanation(
                    assessment_id=assessment.id,
                    user_id=demo_user.id,
                    feature_name=factor['feature_name'],
                    feature_value=factor['feature_value'],
                    shap_value=factor['shap_value'],
                    contribution_level=factor['contribution_level'],
                    direction=factor['direction'],
                    explanation_text=factor['explanation_text'],
                    created_at=rec_date
                )
                db.session.add(shap_entity)

            risk_res = calculate_future_risk(features_dict, pred_res)
            future_risk = FutureRisk(
                assessment_id=assessment.id,
                user_id=demo_user.id,
                risk_level=risk_res['risk_level'],
                risk_score=risk_res['risk_score'],
                risk_trajectory="Improving" if step['days_ago'] < 20 else "Stable",
                primary_risk_factors='; '.join(risk_res['primary_risk_factors']),
                preventive_guidance=risk_res['preventive_guidance'],
                created_at=rec_date
            )
            db.session.add(future_risk)

            rec_res = generate_personalized_recommendations(features_dict, pred_res)
            for item in rec_res['immediate_actions'][:2]:
                rec_entity = Recommendation(
                    assessment_id=assessment.id,
                    user_id=demo_user.id,
                    category='Immediate Actions',
                    recommendation_text=item['text'],
                    priority='High',
                    created_at=rec_date
                )
                db.session.add(rec_entity)

            progress = ProgressRecord(
                user_id=demo_user.id,
                assessment_id=assessment.id,
                record_date=rec_date,
                health_score=score_res['overall_score'],
                sleep_duration=assessment.sleep_duration,
                quality_of_sleep=assessment.quality_of_sleep,
                stress_level=assessment.stress_level,
                physical_activity_level=assessment.physical_activity_level,
                daily_steps=assessment.daily_steps,
                predicted_disorder=pred_res['prediction'],
                risk_level=risk_res['risk_level']
            )
            db.session.add(progress)
            latest_assessment = assessment

        # Initialize or update Digital Twin
        twin = DigitalTwin(
            user_id=demo_user.id,
            current_sleep_duration=latest_assessment.sleep_duration,
            current_quality_of_sleep=latest_assessment.quality_of_sleep,
            current_stress_level=latest_assessment.stress_level,
            current_activity_level=latest_assessment.physical_activity_level,
            current_daily_steps=latest_assessment.daily_steps,
            current_bmi_category=latest_assessment.bmi_category,
            current_blood_pressure=latest_assessment.blood_pressure,
            current_heart_rate=latest_assessment.heart_rate,
            current_predicted_disorder="None",
            current_confidence=0.94,
            current_health_score=88.5,
            current_future_risk="Low",
            total_assessments=len(sample_timeline),
            baseline_health_score=42.0,
            last_assessment_date=latest_assessment.created_at,
            updated_at=datetime.now(timezone.utc)
        )
        db.session.add(twin)
        db.session.commit()

        print(f"Successfully seeded {len(sample_timeline)} DEMO historical records.")
        print("Demo Account Credentials:")
        print("  Email:    demo@sleepwellness.ai")
        print("  Password: password123")
        print("=" * 60)

if __name__ == '__main__':
    seed_demo_data()
