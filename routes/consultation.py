import uuid
import json
from datetime import datetime, timezone
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from utils.helpers import login_required, get_current_user
from models.db_models import db, User, DigitalTwin, SleepAssessment, Prediction, ShapExplanation, FacialAssessment, DoctorConsultation
from services.digital_twin_service import DigitalTwinService

consultation_bp = Blueprint('consultation', __name__)

AVAILABLE_SPECIALISTS = [
    {
        'id': 'spec-1',
        'name': 'Dr. Marcus Vance, MD, DABSM',
        'title': 'Board-Certified Somnologist & Pulmonologist',
        'specialty': 'Sleep Apnea & Airway Diagnostics',
        'hospital': 'Sleep Wellness Institute & Stanford Medicine Affiliate',
        'experience': '16+ years experience',
        'rating': 4.9,
        'reviews_count': 142,
        'consultation_fee': '$150 (Insurance Accepted)',
        'focus_condition': 'Sleep Apnea',
        'bio': 'Specializes in obstructive sleep apnea, nocturnal hypoxia desaturation, and CPAP compliance optimization.',
        'available_slots': ['Today at 4:30 PM', 'Tomorrow at 10:00 AM', 'Tomorrow at 2:15 PM'],
        'avatar_initials': 'MV',
        'color': 'primary'
    },
    {
        'id': 'spec-2',
        'name': 'Dr. Sarah Jenkins, Ph.D., CBSM',
        'title': 'Behavioral Sleep Medicine Specialist',
        'specialty': 'Chronic Insomnia & CBT-I Therapy',
        'hospital': 'Cognitive Sleep Health Center, Boston',
        'experience': '12+ years experience',
        'rating': 4.95,
        'reviews_count': 189,
        'consultation_fee': '$140 (CBT-I Eligible)',
        'focus_condition': 'Insomnia',
        'bio': 'Pioneer in non-pharmacological Cognitive Behavioral Therapy for Insomnia (CBT-I), sleep latency reduction, and circadian realignment.',
        'available_slots': ['Tomorrow at 11:30 AM', 'Wednesday at 3:00 PM', 'Thursday at 5:00 PM'],
        'avatar_initials': 'SJ',
        'color': 'warning'
    },
    {
        'id': 'spec-3',
        'name': 'Dr. Rajesh Kothari, MD, DM',
        'title': 'Clinical Neurologist & Sleep Physician',
        'specialty': 'Neurological Sleep Architecture & RLS',
        'hospital': 'Metro Neuro-Sleep Clinic',
        'experience': '18+ years experience',
        'rating': 4.88,
        'reviews_count': 98,
        'consultation_fee': '$160 (Neuro-Sleep Accredited)',
        'focus_condition': 'Sleep Apnea',
        'bio': 'Expert in parasomnias, central sleep apnea, REM sleep behavior disorders, and neuromuscular nocturnal monitoring.',
        'available_slots': ['Today at 6:00 PM', 'Thursday at 1:00 PM', 'Friday at 11:00 AM'],
        'avatar_initials': 'RK',
        'color': 'info'
    },
    {
        'id': 'spec-4',
        'name': 'Dr. Elena Rostova, MD',
        'title': 'Preventive Sleep & Lifestyle Physician',
        'specialty': 'Circadian Rhythm & Shift-Work Sleep Fatigue',
        'hospital': 'Global Circadian Wellness Center',
        'experience': '10+ years experience',
        'rating': 4.92,
        'reviews_count': 114,
        'consultation_fee': '$135 (Telehealth Certified)',
        'focus_condition': 'None',
        'bio': 'Focuses on occupational fatigue, digital twin biometric tracking, sleep debt recovery, and preventative longevity medicine.',
        'available_slots': ['Tomorrow at 9:00 AM', 'Wednesday at 1:30 PM', 'Friday at 4:00 PM'],
        'avatar_initials': 'ER',
        'color': 'success'
    }
]


@consultation_bp.route('/consultation', methods=['GET'])
@login_required
def consultation_view():
    """
    Renders the Doctor Consultation & Telehealth Portal.
    """
    user_id = session.get('user_id')
    user = get_current_user()
    twin_data = DigitalTwinService.get_digital_twin_state(user_id)

    # Fetch user's consultation bookings
    consultations = DoctorConsultation.query.filter_by(user_id=user_id)\
        .order_by(DoctorConsultation.appointment_date.desc(), DoctorConsultation.created_at.desc()).all()

    # Determine recommended specialist based on user's current disorder
    current_disorder = twin_data['twin']['current_predicted_disorder'] if twin_data and twin_data.get('twin') else 'None'
    recommended_specialist = None
    if current_disorder == 'Sleep Apnea':
        recommended_specialist = AVAILABLE_SPECIALISTS[0] # Dr. Marcus Vance
    elif current_disorder == 'Insomnia':
        recommended_specialist = AVAILABLE_SPECIALISTS[1] # Dr. Sarah Jenkins
    else:
        recommended_specialist = AVAILABLE_SPECIALISTS[3] # Dr. Elena Rostova

    # Latest assessment for clinical context
    latest_assessment = SleepAssessment.query.filter_by(user_id=user_id)\
        .order_by(SleepAssessment.created_at.desc()).first()

    return render_template(
        'consultation.html',
        user=user,
        twin_data=twin_data,
        specialists=AVAILABLE_SPECIALISTS,
        recommended_specialist=recommended_specialist,
        consultations=[c.to_dict() for c in consultations],
        latest_assessment=latest_assessment
    )


@consultation_bp.route('/api/consultation/book', methods=['POST'])
@login_required
def book_consultation():
    """
    Books a virtual doctor consultation and snapshots current Digital Twin state.
    """
    user_id = session.get('user_id')
    user = get_current_user()
    data = request.get_json() if request.is_json else request.form.to_dict()

    doctor_name = data.get('doctor_name', 'Dr. Marcus Vance, MD, DABSM')
    specialty = data.get('specialty', 'Sleep Medicine Specialist')
    hospital_clinic = data.get('hospital_clinic', 'Sleep Wellness Telehealth Network')
    consultation_type = data.get('consultation_type', 'Virtual Video Teleconsultation')
    time_slot = data.get('time_slot', 'Tomorrow at 10:00 AM')
    patient_notes = data.get('patient_notes', '')

    # Snapshot current Digital Twin vitals & ML prediction
    twin_data = DigitalTwinService.get_digital_twin_state(user_id)
    snapshot = {}
    if twin_data and twin_data.get('twin'):
        t = twin_data['twin']
        snapshot = {
            'predicted_disorder': t.get('current_predicted_disorder', 'Pending'),
            'confidence': t.get('current_confidence', 0.0),
            'health_score': t.get('current_health_score', 0.0),
            'future_risk': t.get('current_future_risk', 'Unassessed'),
            'sleep_duration': t.get('current_sleep_duration', 0.0),
            'quality_of_sleep': t.get('current_quality_of_sleep', 0),
            'blood_pressure': t.get('current_blood_pressure', '120/80'),
            'bmi_category': t.get('current_bmi_category', 'Normal'),
            'daily_steps': t.get('current_daily_steps', 0),
            'latest_facial_score': t.get('latest_facial_score'),
            'latest_facial_level': t.get('latest_facial_level'),
            'user_name': user.full_name,
            'user_age': user.age or 35,
            'user_gender': user.gender or 'Not Specified'
        }

    # Generate dedicated encrypted room link
    room_code = uuid.uuid4().hex[:8]
    meeting_link = f"https://meet.sleepwellness.ai/room-sw-{room_code}"

    # Appointment date
    appointment_date = datetime.now(timezone.utc)

    consultation = DoctorConsultation(
        user_id=user_id,
        doctor_name=doctor_name,
        specialty=specialty,
        hospital_clinic=hospital_clinic,
        consultation_type=consultation_type,
        appointment_date=appointment_date,
        time_slot=time_slot,
        status='Confirmed',
        patient_notes=patient_notes,
        digital_twin_snapshot=json.dumps(snapshot),
        meeting_link=meeting_link
    )

    db.session.add(consultation)
    db.session.commit()

    return jsonify({
        'status': 'success',
        'message': f'Consultation with {doctor_name} has been confirmed!',
        'consultation': consultation.to_dict()
    })


@consultation_bp.route('/api/consultation/cancel/<int:consultation_id>', methods=['POST'])
@login_required
def cancel_consultation(consultation_id):
    """
    Cancels an active doctor consultation.
    """
    user_id = session.get('user_id')
    consultation = DoctorConsultation.query.filter_by(id=consultation_id, user_id=user_id).first()
    if not consultation:
        return jsonify({'status': 'error', 'message': 'Consultation record not found'}), 404

    consultation.status = 'Cancelled'
    db.session.commit()

    return jsonify({
        'status': 'success',
        'message': 'Consultation appointment has been cancelled.',
        'consultation_id': consultation_id
    })
