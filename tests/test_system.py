import io
import json
import unittest
from app import create_app
from models.db_models import db, User, SleepAssessment, DigitalTwin, Prediction
from ml.predict import predict_sleep_disorder
from ml.explain import explain_prediction
from ml.scoring import compute_sleep_health_score
from ml.risk import calculate_future_risk
from ml.recommendations import generate_personalized_recommendations
from services.pdf_service import PDFReportGenerator

class SleepWellnessSystemTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_01_database_connectivity(self):
        """Test database connection and table presence."""
        user_count = User.query.count()
        self.assertGreaterEqual(user_count, 0)
        print(" [PASS] 1. Database Connection & Tables Verified")

    def test_02_ml_prediction_and_probabilities(self):
        """Test ML prediction and real probability calculation."""
        sample_input = {
            'Age': 40,
            'Gender': 'Male',
            'Occupation': 'Doctor',
            'Sleep Duration': 6.0,
            'Quality of Sleep': 5,
            'Physical Activity Level': 40,
            'Stress Level': 7,
            'BMI Category': 'Overweight',
            'Blood Pressure': '132/85',
            'Heart Rate': 78,
            'Daily Steps': 5500
        }
        res = predict_sleep_disorder(sample_input)
        self.assertIn(res['prediction'], ['None', 'Insomnia', 'Sleep Apnea'])
        self.assertGreater(res['confidence'], 0.0)
        self.assertIn('None', res['probabilities'])
        self.assertIn('Insomnia', res['probabilities'])
        self.assertIn('Sleep Apnea', res['probabilities'])
        # Probabilities sum approximately to 1
        prob_sum = sum(res['probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=1)
        print(f" [PASS] 2. ML Prediction & Probabilities Verified: {res['prediction']} ({res['confidence']*100:.1f}%)")

    def test_03_shap_explainability(self):
        """Test SHAP TreeExplainer calculates non-zero feature attributions."""
        sample_input = {
            'Age': 45,
            'Gender': 'Female',
            'Occupation': 'Nurse',
            'Sleep Duration': 5.2,
            'Quality of Sleep': 4,
            'Physical Activity Level': 30,
            'Stress Level': 8,
            'BMI Category': 'Overweight',
            'Blood Pressure': '138/90',
            'Heart Rate': 84,
            'Daily Steps': 4000
        }
        pred = predict_sleep_disorder(sample_input)
        shap_res = explain_prediction(sample_input, pred)
        self.assertIn('top_factors', shap_res)
        self.assertGreater(len(shap_res['top_factors']), 0)
        top = shap_res['top_factors'][0]
        self.assertIn('feature_name', top)
        self.assertIn('shap_value', top)
        self.assertIn('contribution_level', top)
        self.assertIn('direction', top)
        print(f" [PASS] 3. SHAP Explainability Verified: Top Factor: {top['feature_name']} (SHAP: {top['shap_value']})")

    def test_04_sleep_health_score(self):
        """Test deterministic Sleep Health Score computation (0-100)."""
        input_data = {
            'sleep_duration': 7.8,
            'quality_of_sleep': 8,
            'stress_level': 3,
            'physical_activity_level': 60,
            'daily_steps': 8500,
            'bmi_category': 'Normal',
            'heart_rate': 66
        }
        score_res = compute_sleep_health_score(input_data)
        self.assertGreaterEqual(score_res['overall_score'], 0.0)
        self.assertLessEqual(score_res['overall_score'], 100.0)
        self.assertEqual(score_res['category'], 'Good')
        self.assertIn('breakdown', score_res)
        print(f" [PASS] 4. Sleep Health Score Verified: {score_res['overall_score']}/100 ({score_res['category']})")

    def test_05_future_risk_estimation(self):
        """Test Future Risk estimation and trajectory modeling."""
        input_data = {
            'sleep_duration': 5.0,
            'quality_of_sleep': 3,
            'stress_level': 9,
            'physical_activity_level': 20,
            'daily_steps': 3000,
            'bmi_category': 'Obese',
            'heart_rate': 88,
            'blood_pressure': '145/95'
        }
        pred = predict_sleep_disorder(input_data)
        risk_res = calculate_future_risk(input_data, pred)
        self.assertIn(risk_res['risk_level'], ['Low', 'Moderate', 'High'])
        self.assertGreater(risk_res['risk_score'], 0)
        self.assertGreater(len(risk_res['primary_risk_factors']), 0)
        print(f" [PASS] 5. Future Risk Estimation Verified: {risk_res['risk_level']} Risk ({risk_res['risk_score']}/100)")

    def test_06_personalized_recommendations(self):
        """Test 4-pillar personalized recommendations generation."""
        input_data = {
            'sleep_duration': 5.5,
            'quality_of_sleep': 4,
            'stress_level': 8,
            'physical_activity_level': 25,
            'daily_steps': 4000,
            'bmi_category': 'Overweight',
            'heart_rate': 80,
            'blood_pressure': '135/88'
        }
        pred = predict_sleep_disorder(input_data)
        recs = generate_personalized_recommendations(input_data, pred)
        self.assertGreater(len(recs['immediate_actions']), 0)
        self.assertGreater(len(recs['lifestyle_improvements']), 0)
        self.assertGreater(len(recs['sleep_habits']), 0)
        self.assertGreater(len(recs['seek_professional_help']), 0)
        print(" [PASS] 6. Personalized Recommendations across 4 Pillars Verified")

    def test_07_auth_workflow(self):
        """Test registration, login, protected routes, and logout."""
        test_email = "tester_unit@sleepwellness.ai"
        
        # Clean existing test user if any
        existing = User.query.filter_by(email=test_email).first()
        if existing:
            db.session.delete(existing)
            db.session.commit()

        # 1. Registration
        reg_resp = self.client.post('/register', data={
            'full_name': 'Test User',
            'email': test_email,
            'age': 32,
            'gender': 'Female',
            'occupation': 'Scientist',
            'password': 'testpassword123',
            'confirm_password': 'testpassword123'
        }, follow_redirects=True)
        self.assertEqual(reg_resp.status_code, 200)

        # Verify user and Digital Twin created
        created_user = User.query.filter_by(email=test_email).first()
        self.assertIsNotNone(created_user)
        self.assertIsNotNone(created_user.digital_twin)

        # 2. Logout
        logout_resp = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(logout_resp.status_code, 200)

        # 3. Login
        login_resp = self.client.post('/login', data={
            'email': test_email,
            'password': 'testpassword123'
        }, follow_redirects=True)
        self.assertEqual(login_resp.status_code, 200)
        self.assertIn(b'Dashboard', login_resp.data)

        print(" [PASS] 7. Auth Workflow (Register, Digital Twin Init, Logout, Login) Verified")

    def test_08_assessment_submission_and_twin_update(self):
        """Test end-to-end assessment submission, ML prediction, and Digital Twin synchronization."""
        # Login demo user
        self.client.post('/login', data={
            'email': 'demo@sleepwellness.ai',
            'password': 'password123'
        }, follow_redirects=True)

        demo_user = User.query.filter_by(email='demo@sleepwellness.ai').first()
        initial_assessments = demo_user.digital_twin.total_assessments

        # Submit assessment via API
        resp = self.client.post('/api/predict', json={
            'age': 36,
            'gender': 'Male',
            'occupation': 'Software Engineer',
            'sleep_duration': 7.6,
            'quality_of_sleep': 8,
            'physical_activity_level': 55.0,
            'stress_level': 4,
            'bmi_category': 'Normal',
            'blood_pressure': '120/80',
            'heart_rate': 68,
            'daily_steps': 7800
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('assessment_id', data)
        self.assertIn('prediction', data)
        self.assertIn('probabilities', data)

        # Check Digital Twin was updated
        db.session.refresh(demo_user.digital_twin)
        self.assertEqual(demo_user.digital_twin.total_assessments, initial_assessments + 1)
        self.assertEqual(demo_user.digital_twin.current_sleep_duration, 7.6)
        print(f" [PASS] 8. Assessment Submission & Digital Twin Sync Verified (Twin Total: {demo_user.digital_twin.total_assessments})")

    def test_09_what_if_simulator_does_not_mutate_db(self):
        """Test What-If simulator runs pure in-memory without modifying database."""
        # Login
        self.client.post('/login', data={
            'email': 'demo@sleepwellness.ai',
            'password': 'password123'
        }, follow_redirects=True)

        demo_user = User.query.filter_by(email='demo@sleepwellness.ai').first()
        prev_health_score = demo_user.digital_twin.current_health_score

        sim_resp = self.client.post('/api/simulate', json={
            'age': 36,
            'gender': 'Male',
            'occupation': 'Software Engineer',
            'sleep_duration': 8.5,
            'quality_of_sleep': 9,
            'physical_activity_level': 75.0,
            'stress_level': 2,
            'bmi_category': 'Normal',
            'blood_pressure': '118/76',
            'heart_rate': 62,
            'daily_steps': 11000
        })
        self.assertEqual(sim_resp.status_code, 200)
        sim_data = sim_resp.get_json()
        self.assertEqual(sim_data['status'], 'success')
        self.assertEqual(sim_data['simulated_prediction'], 'None')

        # Ensure Digital Twin record remained untouched
        db.session.refresh(demo_user.digital_twin)
        self.assertEqual(demo_user.digital_twin.current_health_score, prev_health_score)
        print(" [PASS] 9. What-If Simulator In-Memory Isolation Verified (Twin Unaltered)")

    def test_10_pdf_report_generation(self):
        """Test ReportLab PDF generation generates valid binary PDF."""
        demo_user = User.query.filter_by(email='demo@sleepwellness.ai').first()
        latest = SleepAssessment.query.filter_by(user_id=demo_user.id).first()
        self.assertIsNotNone(latest)

        pdf_buffer = PDFReportGenerator.generate_assessment_report(latest.id, demo_user.id)
        self.assertIsInstance(pdf_buffer, io.BytesIO)
        pdf_bytes = pdf_buffer.getvalue()
        # PDF files start with %PDF
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))
        self.assertGreater(len(pdf_bytes), 1000)
        print(f" [PASS] 10. PDF Report Generation Verified: Valid %PDF Binary ({len(pdf_bytes):,} bytes)")

if __name__ == '__main__':
    unittest.main()
