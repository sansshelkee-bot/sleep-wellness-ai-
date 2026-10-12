from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

def current_time():
    return datetime.now(timezone.utc)

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    occupation = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=current_time)
    updated_at = db.Column(db.DateTime, default=current_time, onupdate=current_time)

    # Relationships
    assessments = db.relationship('SleepAssessment', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    digital_twin = db.relationship('DigitalTwin', backref='user', uselist=False, cascade='all, delete-orphan')
    predictions = db.relationship('Prediction', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    recommendations = db.relationship('Recommendation', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    progress_records = db.relationship('ProgressRecord', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    facial_assessments = db.relationship('FacialAssessment', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    consultations = db.relationship('DoctorConsultation', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'full_name': self.full_name,
            'email': self.email,
            'age': self.age,
            'gender': self.gender,
            'occupation': self.occupation,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }


class SleepAssessment(db.Model):
    __tablename__ = 'sleep_assessments'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    occupation = db.Column(db.String(100), nullable=False)
    sleep_duration = db.Column(db.Float, nullable=False)
    quality_of_sleep = db.Column(db.Integer, nullable=False)
    physical_activity_level = db.Column(db.Float, nullable=False)
    stress_level = db.Column(db.Integer, nullable=False)
    bmi_category = db.Column(db.String(50), nullable=False)
    blood_pressure = db.Column(db.String(20), nullable=False)
    systolic_bp = db.Column(db.Integer, nullable=True)
    diastolic_bp = db.Column(db.Integer, nullable=True)
    heart_rate = db.Column(db.Integer, nullable=False)
    daily_steps = db.Column(db.Integer, nullable=False)
    is_demo = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=current_time, index=True)

    # Relationships
    prediction = db.relationship('Prediction', backref='assessment', uselist=False, cascade='all, delete-orphan')
    recommendations = db.relationship('Recommendation', backref='assessment', lazy='dynamic', cascade='all, delete-orphan')
    shap_explanations = db.relationship('ShapExplanation', backref='assessment', lazy='dynamic', cascade='all, delete-orphan')
    future_risk = db.relationship('FutureRisk', backref='assessment', uselist=False, cascade='all, delete-orphan')
    progress_record = db.relationship('ProgressRecord', backref='assessment', uselist=False, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'age': self.age,
            'gender': self.gender,
            'occupation': self.occupation,
            'sleep_duration': round(self.sleep_duration, 1),
            'quality_of_sleep': self.quality_of_sleep,
            'physical_activity_level': self.physical_activity_level,
            'stress_level': self.stress_level,
            'bmi_category': self.bmi_category,
            'blood_pressure': self.blood_pressure,
            'systolic_bp': self.systolic_bp,
            'diastolic_bp': self.diastolic_bp,
            'heart_rate': self.heart_rate,
            'daily_steps': self.daily_steps,
            'is_demo': self.is_demo,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None
        }


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey('sleep_assessments.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    predicted_disorder = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    prob_none = db.Column(db.Float, nullable=False)
    prob_insomnia = db.Column(db.Float, nullable=False)
    prob_apnea = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(50), default='CatBoost v1.2')
    created_at = db.Column(db.DateTime, default=current_time)

    def to_dict(self):
        return {
            'id': self.id,
            'assessment_id': self.assessment_id,
            'predicted_disorder': self.predicted_disorder,
            'confidence': round(self.confidence * 100, 1),
            'probabilities': {
                'None': round(self.prob_none * 100, 1),
                'Insomnia': round(self.prob_insomnia * 100, 1),
                'Sleep Apnea': round(self.prob_apnea * 100, 1)
            },
            'model_version': self.model_version,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None
        }


class DigitalTwin(db.Model):
    __tablename__ = 'digital_twins'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False)
    current_sleep_duration = db.Column(db.Float, default=0.0)
    current_quality_of_sleep = db.Column(db.Integer, default=0)
    current_stress_level = db.Column(db.Integer, default=0)
    current_activity_level = db.Column(db.Float, default=0.0)
    current_daily_steps = db.Column(db.Integer, default=0)
    current_bmi_category = db.Column(db.String(50), default='Normal')
    current_blood_pressure = db.Column(db.String(20), default='120/80')
    current_heart_rate = db.Column(db.Integer, default=70)
    current_predicted_disorder = db.Column(db.String(50), default='Pending Assessment')
    current_confidence = db.Column(db.Float, default=0.0)
    current_health_score = db.Column(db.Float, default=0.0)
    current_future_risk = db.Column(db.String(20), default='Unassessed')
    total_assessments = db.Column(db.Integer, default=0)
    baseline_health_score = db.Column(db.Float, default=0.0)
    latest_facial_score = db.Column(db.Float, nullable=True)
    latest_facial_level = db.Column(db.String(20), nullable=True)
    last_facial_assessment_date = db.Column(db.DateTime, nullable=True)
    last_assessment_date = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, default=current_time, onupdate=current_time)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'current_sleep_duration': round(self.current_sleep_duration, 1) if self.current_sleep_duration else 0.0,
            'current_quality_of_sleep': self.current_quality_of_sleep or 0,
            'current_stress_level': self.current_stress_level or 0,
            'current_activity_level': self.current_activity_level or 0,
            'current_daily_steps': self.current_daily_steps or 0,
            'current_bmi_category': self.current_bmi_category or 'Normal',
            'current_blood_pressure': self.current_blood_pressure or '120/80',
            'current_heart_rate': self.current_heart_rate or 70,
            'current_predicted_disorder': self.current_predicted_disorder or 'Pending Assessment',
            'current_confidence': round(self.current_confidence * 100, 1) if self.current_confidence else 0.0,
            'current_health_score': round(self.current_health_score, 1) if self.current_health_score else 0.0,
            'current_future_risk': self.current_future_risk or 'Unassessed',
            'total_assessments': self.total_assessments or 0,
            'baseline_health_score': round(self.baseline_health_score, 1) if self.baseline_health_score else 0.0,
            'latest_facial_score': round(self.latest_facial_score, 1) if self.latest_facial_score is not None else None,
            'latest_facial_level': self.latest_facial_level or 'Unassessed',
            'last_facial_assessment_date': self.last_facial_assessment_date.strftime('%Y-%m-%d %H:%M') if self.last_facial_assessment_date else None,
            'last_assessment_date': self.last_assessment_date.strftime('%Y-%m-%d %H:%M') if self.last_assessment_date else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M') if self.updated_at else None
        }


class Recommendation(db.Model):
    __tablename__ = 'recommendations'

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey('sleep_assessments.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    category = db.Column(db.String(50), nullable=False)
    recommendation_text = db.Column(db.Text, nullable=False)
    priority = db.Column(db.String(20), default='Medium')
    created_at = db.Column(db.DateTime, default=current_time)

    def to_dict(self):
        return {
            'id': self.id,
            'category': self.category,
            'recommendation_text': self.recommendation_text,
            'priority': self.priority
        }


class ShapExplanation(db.Model):
    __tablename__ = 'shap_explanations'

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey('sleep_assessments.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    feature_name = db.Column(db.String(100), nullable=False)
    feature_value = db.Column(db.String(100), nullable=True)
    shap_value = db.Column(db.Float, nullable=False)
    contribution_level = db.Column(db.String(20), nullable=False)
    direction = db.Column(db.String(30), nullable=False)
    explanation_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=current_time)

    def to_dict(self):
        return {
            'feature_name': self.feature_name,
            'feature_value': self.feature_value,
            'shap_value': round(self.shap_value, 4),
            'contribution_level': self.contribution_level,
            'direction': self.direction,
            'explanation_text': self.explanation_text
        }


class FutureRisk(db.Model):
    __tablename__ = 'future_risk'

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey('sleep_assessments.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    risk_level = db.Column(db.String(20), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    risk_trajectory = db.Column(db.String(50), default='Stable')
    primary_risk_factors = db.Column(db.Text, nullable=True)
    preventive_guidance = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=current_time)

    def to_dict(self):
        return {
            'risk_level': self.risk_level,
            'risk_score': round(self.risk_score, 1),
            'risk_trajectory': self.risk_trajectory,
            'primary_risk_factors': self.primary_risk_factors.split(';') if self.primary_risk_factors else [],
            'preventive_guidance': self.preventive_guidance
        }


class ProgressRecord(db.Model):
    __tablename__ = 'progress_records'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey('sleep_assessments.id', ondelete='CASCADE'), nullable=False)
    record_date = db.Column(db.DateTime, default=current_time, index=True)
    health_score = db.Column(db.Float, nullable=False)
    sleep_duration = db.Column(db.Float, nullable=False)
    quality_of_sleep = db.Column(db.Integer, nullable=False)
    stress_level = db.Column(db.Integer, nullable=False)
    physical_activity_level = db.Column(db.Float, nullable=False)
    daily_steps = db.Column(db.Integer, nullable=False)
    predicted_disorder = db.Column(db.String(50), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'date': self.record_date.strftime('%Y-%m-%d') if self.record_date else '',
            'health_score': round(self.health_score, 1),
            'sleep_duration': round(self.sleep_duration, 1),
            'quality_of_sleep': self.quality_of_sleep,
            'stress_level': self.stress_level,
            'physical_activity_level': self.physical_activity_level,
            'daily_steps': self.daily_steps,
            'predicted_disorder': self.predicted_disorder,
            'risk_level': self.risk_level
        }


class FacialAssessment(db.Model):
    __tablename__ = 'facial_assessments'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    assessment_date = db.Column(db.DateTime, default=current_time, index=True)
    sleepiness_score = db.Column(db.Float, nullable=False)   # 0 to 100
    sleepiness_level = db.Column(db.String(20), nullable=False) # 'Low', 'Moderate', 'High'
    
    # Quantitative Biometric Telemetry
    blink_count = db.Column(db.Integer, default=0)
    blink_rate_per_min = db.Column(db.Float, default=0.0)
    avg_ear = db.Column(db.Float, default=0.0)
    perclos_percentage = db.Column(db.Float, default=0.0)
    yawn_count = db.Column(db.Integer, default=0)
    max_mar = db.Column(db.Float, default=0.0)
    head_nod_count = db.Column(db.Integer, default=0)
    head_movement_detected = db.Column(db.Boolean, default=False)
    duration_seconds = db.Column(db.Float, default=15.0)

    # Explainability & Recommendations
    observed_indicators = db.Column(db.Text, nullable=True)   # Semicolon-delimited list of detected indicators
    contributing_factors = db.Column(db.Text, nullable=True)  # JSON string of factor contributions
    recommendation = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=current_time)

    def to_dict(self):
        import json
        factors = []
        if self.contributing_factors:
            try:
                factors = json.loads(self.contributing_factors)
            except Exception:
                factors = []

        return {
            'id': self.id,
            'user_id': self.user_id,
            'date': self.assessment_date.strftime('%Y-%m-%d %H:%M') if self.assessment_date else '',
            'sleepiness_score': round(self.sleepiness_score, 1),
            'sleepiness_level': self.sleepiness_level,
            'blink_count': self.blink_count,
            'blink_rate_per_min': round(self.blink_rate_per_min, 1),
            'avg_ear': round(self.avg_ear, 3),
            'perclos_percentage': round(self.perclos_percentage, 1),
            'yawn_count': self.yawn_count,
            'max_mar': round(self.max_mar, 3),
            'head_nod_count': self.head_nod_count,
            'head_movement_detected': self.head_movement_detected,
            'duration_seconds': round(self.duration_seconds, 1),
            'observed_indicators': [ind.strip() for ind in self.observed_indicators.split(';') if ind.strip()] if self.observed_indicators else [],
            'contributing_factors': factors,
            'recommendation': self.recommendation
        }


class DoctorConsultation(db.Model):
    __tablename__ = 'doctor_consultations'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    doctor_name = db.Column(db.String(100), nullable=False)
    specialty = db.Column(db.String(100), nullable=False)
    hospital_clinic = db.Column(db.String(150), nullable=False)
    consultation_type = db.Column(db.String(50), default='Virtual Video Teleconsultation')
    appointment_date = db.Column(db.DateTime, nullable=False)
    time_slot = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(30), default='Confirmed')  # 'Confirmed', 'Completed', 'Cancelled'
    patient_notes = db.Column(db.Text, nullable=True)
    digital_twin_snapshot = db.Column(db.Text, nullable=True)  # JSON summary of twin at booking
    meeting_link = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=current_time)

    def to_dict(self):
        import json
        snapshot = {}
        if self.digital_twin_snapshot:
            try:
                snapshot = json.loads(self.digital_twin_snapshot)
            except Exception:
                snapshot = {}

        return {
            'id': self.id,
            'user_id': self.user_id,
            'doctor_name': self.doctor_name,
            'specialty': self.specialty,
            'hospital_clinic': self.hospital_clinic,
            'consultation_type': self.consultation_type,
            'appointment_date': self.appointment_date.strftime('%Y-%m-%d') if self.appointment_date else '',
            'time_slot': self.time_slot,
            'status': self.status,
            'patient_notes': self.patient_notes or '',
            'digital_twin_snapshot': snapshot,
            'meeting_link': self.meeting_link or '',
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else ''
        }

