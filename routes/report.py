from flask import Blueprint, send_file, flash, redirect, url_for, session
from models.db_models import SleepAssessment
from services.pdf_service import PDFReportGenerator
from utils.helpers import login_required

report_bp = Blueprint('report', __name__)

@report_bp.route('/report/<int:assessment_id>')
@login_required
def download_report(assessment_id):
    user_id = session.get('user_id')
    try:
        pdf_buffer = PDFReportGenerator.generate_assessment_report(assessment_id, user_id)
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"Sleep_Wellness_Report_Assessment_{assessment_id}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        flash(f"Error generating PDF report: {e}", "danger")
        return redirect(url_for('dashboard.dashboard_view'))


@report_bp.route('/report/latest')
@login_required
def download_latest_report():
    user_id = session.get('user_id')
    latest = SleepAssessment.query.filter_by(user_id=user_id).order_by(SleepAssessment.created_at.desc()).first()
    if not latest:
        flash("No assessments found. Please complete an assessment first.", "warning")
        return redirect(url_for('prediction.assessment_form'))
    return download_report(latest.id)
