from datetime import datetime, timezone
from models.db_models import (
    db, User, SleepAssessment, Prediction, DigitalTwin,
    Recommendation, ShapExplanation, FutureRisk, ProgressRecord
)
from ml.predict import predict_sleep_disorder
from ml.explain import explain_prediction
from ml.scoring import compute_sleep_health_score
from ml.risk import calculate_future_risk
from ml.recommendations import generate_personalized_recommendations
from ml.preprocess import parse_blood_pressure


class DigitalTwinService:
    """
    Central orchestration engine for the Digital Twin.
    Maintains continuous virtual synchronization with user's physiological and lifestyle data.
    """

    @staticmethod
    def process_new_assessment(user_id, input_data, is_demo=False):
        """
        Executes the entire end-to-end AI pipeline:
        Assessment -> ML Prediction -> SHAP -> Health Score -> Risk -> Recommendations -> Digital Twin Update
        """
        user = db.session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} does not exist.")

        # 1. Parse Blood Pressure
        bp_raw = str(input_data.get('blood_pressure', '120/80'))
        systolic, diastolic = parse_blood_pressure(bp_raw)

        # 2. Create and commit SleepAssessment record
        assessment = SleepAssessment(
            user_id=user_id,
            age=int(input_data.get('age', user.age or 35)),
            gender=str(input_data.get('gender', user.gender or 'Male')).title(),
            occupation=str(input_data.get('occupation', user.occupation or 'Other')),
            sleep_duration=float(input_data.get('sleep_duration', 7.0)),
            quality_of_sleep=int(input_data.get('quality_of_sleep', 7)),
            physical_activity_level=float(input_data.get('physical_activity_level', 45.0)),
            stress_level=int(input_data.get('stress_level', 5)),
            bmi_category=str(input_data.get('bmi_category', 'Normal')).strip().title(),
            blood_pressure=f"{systolic}/{diastolic}",
            systolic_bp=systolic,
            diastolic_bp=diastolic,
            heart_rate=int(input_data.get('heart_rate', 70)),
            daily_steps=int(input_data.get('daily_steps', 6000)),
            is_demo=is_demo,
            created_at=datetime.now(timezone.utc)
        )
        db.session.add(assessment)
        db.session.flush()  # Retrieve assessment.id

        # 3. Machine Learning Prediction & Probabilities
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

        pred_result = predict_sleep_disorder(features_dict)

        # Save Prediction entity
        prediction_record = Prediction(
            assessment_id=assessment.id,
            user_id=user_id,
            predicted_disorder=pred_result['prediction'],
            confidence=pred_result['confidence'],
            prob_none=pred_result['probabilities'].get('None', 0.0),
            prob_insomnia=pred_result['probabilities'].get('Insomnia', 0.0),
            prob_apnea=pred_result['probabilities'].get('Sleep Apnea', 0.0),
            model_version='Voting Ensemble (CatBoost+RF+GB)',
            created_at=datetime.now(timezone.utc)
        )
        db.session.add(prediction_record)

        # 4. Compute Sleep Health Score (0-100)
        score_result = compute_sleep_health_score(features_dict, pred_result)
        health_score_val = score_result['overall_score']

        # 5. Calculate Explainable AI (SHAP)
        shap_result = explain_prediction(features_dict, pred_result)
        for factor in shap_result.get('top_factors', [])[:6]:
            shap_entity = ShapExplanation(
                assessment_id=assessment.id,
                user_id=user_id,
                feature_name=factor['feature_name'],
                feature_value=factor['feature_value'],
                shap_value=factor['shap_value'],
                contribution_level=factor['contribution_level'],
                direction=factor['direction'],
                explanation_text=factor['explanation_text'],
                created_at=datetime.now(timezone.utc)
            )
            db.session.add(shap_entity)

        # 6. Fetch previous assessments to calculate trajectory
        past_assessments = [
            {'health_score': p.health_score}
            for p in ProgressRecord.query.filter_by(user_id=user_id).order_by(ProgressRecord.record_date.desc()).limit(5).all()
        ]

        # 7. Compute Future Risk
        risk_result = calculate_future_risk(features_dict, pred_result, past_assessments)
        risk_record = FutureRisk(
            assessment_id=assessment.id,
            user_id=user_id,
            risk_level=risk_result['risk_level'],
            risk_score=risk_result['risk_score'],
            risk_trajectory=risk_result['risk_trajectory'],
            primary_risk_factors='; '.join(risk_result['primary_risk_factors']),
            preventive_guidance=risk_result['preventive_guidance'],
            created_at=datetime.now(timezone.utc)
        )
        db.session.add(risk_record)

        # 8. Generate Personalized Recommendations
        rec_result = generate_personalized_recommendations(features_dict, pred_result)
        for pillar, items in [
            ('Immediate Actions', rec_result['immediate_actions']),
            ('Lifestyle Improvements', rec_result['lifestyle_improvements']),
            ('Sleep Habits', rec_result['sleep_habits']),
            ('When to Seek Professional Help', rec_result['seek_professional_help'])
        ]:
            for item in items:
                rec_entity = Recommendation(
                    assessment_id=assessment.id,
                    user_id=user_id,
                    category=pillar,
                    recommendation_text=item['text'],
                    priority=item.get('priority', 'Medium'),
                    created_at=datetime.now(timezone.utc)
                )
                db.session.add(rec_entity)

        # 9. Store Progress Record for timeseries charts
        progress = ProgressRecord(
            user_id=user_id,
            assessment_id=assessment.id,
            record_date=datetime.now(timezone.utc),
            health_score=health_score_val,
            sleep_duration=assessment.sleep_duration,
            quality_of_sleep=assessment.quality_of_sleep,
            stress_level=assessment.stress_level,
            physical_activity_level=assessment.physical_activity_level,
            daily_steps=assessment.daily_steps,
            predicted_disorder=pred_result['prediction'],
            risk_level=risk_result['risk_level']
        )
        db.session.add(progress)

        # 10. Synchronize / Update User's Digital Twin
        twin = DigitalTwin.query.filter_by(user_id=user_id).first()
        if not twin:
            twin = DigitalTwin(
                user_id=user_id,
                baseline_health_score=health_score_val,
                total_assessments=0
            )
            db.session.add(twin)

        twin.current_sleep_duration = assessment.sleep_duration
        twin.current_quality_of_sleep = assessment.quality_of_sleep
        twin.current_stress_level = assessment.stress_level
        twin.current_activity_level = assessment.physical_activity_level
        twin.current_daily_steps = assessment.daily_steps
        twin.current_bmi_category = assessment.bmi_category
        twin.current_blood_pressure = assessment.blood_pressure
        twin.current_heart_rate = assessment.heart_rate
        twin.current_predicted_disorder = pred_result['prediction']
        twin.current_confidence = pred_result['confidence']
        twin.current_health_score = health_score_val
        twin.current_future_risk = risk_result['risk_level']
        twin.total_assessments = (twin.total_assessments or 0) + 1
        twin.last_assessment_date = datetime.now(timezone.utc)

        db.session.commit()

        return {
            'assessment_id': assessment.id,
            'prediction': pred_result,
            'health_score': score_result,
            'shap_explanations': shap_result,
            'future_risk': risk_result,
            'recommendations': rec_result,
            'digital_twin': twin.to_dict()
        }

    @staticmethod
    def get_digital_twin_state(user_id):
        """
        Retrieves the user's active Digital Twin state with baseline deltas
        and historical milestones.
        """
        twin = DigitalTwin.query.filter_by(user_id=user_id).first()
        user = db.session.get(User, user_id)
        if not user:
            return None

        if not twin:
            # Create unassessed placeholder
            twin = DigitalTwin(
                user_id=user_id,
                baseline_health_score=0.0,
                total_assessments=0
            )
            db.session.add(twin)
            db.session.commit()

        # Calculate delta
        delta_score = 0.0
        if twin.baseline_health_score and twin.current_health_score:
            delta_score = round(twin.current_health_score - twin.baseline_health_score, 1)

        # Recent 5 progress records
        recent_records = ProgressRecord.query.filter_by(user_id=user_id).order_by(ProgressRecord.record_date.desc()).limit(5).all()

        return {
            'twin': twin.to_dict(),
            'user': user.to_dict(),
            'delta_score': delta_score,
            'recent_history': [r.to_dict() for r in recent_records]
        }

    @staticmethod
    def get_progress_data(user_id):
        """
        Returns chronological data for Chart.js.
        """
        records = ProgressRecord.query.filter_by(user_id=user_id).order_by(ProgressRecord.record_date.asc()).all()
        labels = []
        scores = []
        durations = []
        stresses = []
        activities = []
        steps = []
        disorders = []

        for r in records:
            d_str = r.record_date.strftime('%b %d') if r.record_date else ''
            labels.append(d_str)
            scores.append(round(r.health_score, 1))
            durations.append(round(r.sleep_duration, 1))
            stresses.append(r.stress_level)
            activities.append(r.physical_activity_level)
            steps.append(r.daily_steps)
            disorders.append(r.predicted_disorder)

        return {
            'labels': labels,
            'scores': scores,
            'durations': durations,
            'stresses': stresses,
            'activities': activities,
            'steps': steps,
            'disorders': disorders
        }
