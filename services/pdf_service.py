import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from models.db_models import (
    db, SleepAssessment, Prediction, ShapExplanation, FutureRisk, Recommendation, User
)

class PDFReportGenerator:
    """
    Generates high-fidelity clinical/wellness Sleep Health PDF reports
    using ReportLab.
    """

    @staticmethod
    def generate_assessment_report(assessment_id, user_id):
        assessment = SleepAssessment.query.filter_by(id=assessment_id, user_id=user_id).first()
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found for this user.")

        user = db.session.get(User, user_id)
        prediction = Prediction.query.filter_by(assessment_id=assessment.id).first()
        shap_factors = ShapExplanation.query.filter_by(assessment_id=assessment.id).order_by(ShapExplanation.shap_value.desc()).all()
        risk = FutureRisk.query.filter_by(assessment_id=assessment.id).first()
        recommendations = Recommendation.query.filter_by(assessment_id=assessment.id).all()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom styles
        primary_color = colors.HexColor('#1e293b')  # Slate 800
        accent_color = colors.HexColor('#0284c7')   # Sky 600
        muted_color = colors.HexColor('#64748b')    # Slate 500
        light_bg = colors.HexColor('#f8fafc')       # Slate 50

        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=primary_color
        )

        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=13,
            textColor=muted_color
        )

        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=primary_color,
            spaceBefore=10,
            spaceAfter=4
        )

        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=primary_color
        )

        bold_style = ParagraphStyle(
            'Bold',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=12,
            textColor=primary_color
        )

        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor('#94a3b8')
        )

        story = []

        # Header Block
        header_data = [
            [
                Paragraph("<b>INTELLIGENT SLEEP WELLNESS REPORT</b>", title_style),
                Paragraph(f"<b>Report ID:</b> SLP-{assessment.id:05d}<br/><b>Date:</b> {assessment.created_at.strftime('%B %d, %Y')}", subtitle_style)
            ]
        ]
        header_table = Table(header_data, colWidths=[4.0 * inch, 3.2 * inch])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT')
        ]))
        story.append(header_table)
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceBefore=2, spaceAfter=8))

        # Patient Info & Core Summary
        pred_label = prediction.predicted_disorder if prediction else 'Pending'
        confidence_pct = f"{round(prediction.confidence * 100, 1)}%" if prediction else 'N/A'

        user_info_data = [
            [
                Paragraph("<b>Subject:</b>", bold_style), Paragraph(user.full_name, body_style),
                Paragraph("<b>Age / Gender:</b>", bold_style), Paragraph(f"{assessment.age} yrs / {assessment.gender}", body_style)
            ],
            [
                Paragraph("<b>Occupation:</b>", bold_style), Paragraph(assessment.occupation, body_style),
                Paragraph("<b>Assessment Date:</b>", bold_style), Paragraph(assessment.created_at.strftime('%Y-%m-%d %H:%M'), body_style)
            ]
        ]
        user_info_table = Table(user_info_data, colWidths=[1.1 * inch, 2.5 * inch, 1.4 * inch, 2.2 * inch])
        user_info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), light_bg),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(user_info_table)
        story.append(Spacer(1, 10))

        # Executive AI Diagnostics
        story.append(Paragraph("1. AI Diagnostic Prediction & Probabilities", section_heading))
        p_none = f"{round(prediction.prob_none * 100, 1)}%" if prediction else '-'
        p_insomnia = f"{round(prediction.prob_insomnia * 100, 1)}%" if prediction else '-'
        p_apnea = f"{round(prediction.prob_apnea * 100, 1)}%" if prediction else '-'

        risk_level_str = risk.risk_level if risk else 'Unassessed'
        risk_score_str = f"{round(risk.risk_score, 1)} / 100" if risk else 'N/A'

        ai_diag_data = [
            [
                Paragraph("<b>Predicted Condition:</b>", bold_style),
                Paragraph(f"<font color='{accent_color}'><b>{pred_label}</b></font> ({confidence_pct} confidence)", body_style),
                Paragraph("<b>Estimated Future Risk:</b>", bold_style),
                Paragraph(f"<b>{risk_level_str}</b> (Score: {risk_score_str})", body_style)
            ],
            [
                Paragraph("<b>Model Architecture:</b>", bold_style),
                Paragraph("CatBoost Gradient Boosted Ensemble", body_style),
                Paragraph("<b>Risk Trajectory:</b>", bold_style),
                Paragraph(risk.risk_trajectory if risk else 'Stable', body_style)
            ],
            [
                Paragraph("<b>Class Probabilities:</b>", bold_style),
                Paragraph(f"None (Healthy): <b>{p_none}</b> | Insomnia: <b>{p_insomnia}</b> | Sleep Apnea: <b>{p_apnea}</b>", body_style),
                Paragraph("", bold_style), Paragraph("", body_style)
            ]
        ]
        ai_table = Table(ai_diag_data, colWidths=[1.5 * inch, 2.3 * inch, 1.4 * inch, 2.0 * inch])
        ai_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 10))

        # Physiological & Lifestyle Metrics
        story.append(Paragraph("2. Entered Sleep & Physiological Metrics", section_heading))
        metrics_data = [
            [
                Paragraph("<b>Sleep Duration:</b>", bold_style), Paragraph(f"{assessment.sleep_duration} hrs", body_style),
                Paragraph("<b>Sleep Quality:</b>", bold_style), Paragraph(f"{assessment.quality_of_sleep} / 10", body_style),
                Paragraph("<b>Stress Level:</b>", bold_style), Paragraph(f"{assessment.stress_level} / 10", body_style)
            ],
            [
                Paragraph("<b>Physical Activity:</b>", bold_style), Paragraph(f"{assessment.physical_activity_level} min/day", body_style),
                Paragraph("<b>Daily Steps:</b>", bold_style), Paragraph(f"{assessment.daily_steps:,} steps", body_style),
                Paragraph("<b>Heart Rate:</b>", bold_style), Paragraph(f"{assessment.heart_rate} bpm", body_style)
            ],
            [
                Paragraph("<b>BMI Category:</b>", bold_style), Paragraph(assessment.bmi_category, body_style),
                Paragraph("<b>Blood Pressure:</b>", bold_style), Paragraph(assessment.blood_pressure, body_style),
                Paragraph("<b>Systolic/Diastolic:</b>", bold_style), Paragraph(f"{assessment.systolic_bp}/{assessment.diastolic_bp} mmHg", body_style)
            ]
        ]
        metrics_table = Table(metrics_data, colWidths=[1.3 * inch, 1.1 * inch, 1.2 * inch, 1.1 * inch, 1.2 * inch, 1.3 * inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), light_bg),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 10))

        # Explainable AI: Top Contributing SHAP Factors
        story.append(Paragraph("3. Explainable AI: Key Influencing Factors (SHAP Analysis)", section_heading))
        shap_header = [
            Paragraph("<b>Feature</b>", bold_style),
            Paragraph("<b>Reported Value</b>", bold_style),
            Paragraph("<b>Impact Level</b>", bold_style),
            Paragraph("<b>Direction</b>", bold_style),
            Paragraph("<b>Model Influence Insight</b>", bold_style)
        ]
        shap_rows = [shap_header]
        for f in shap_factors[:5]:
            shap_rows.append([
                Paragraph(f.feature_name, body_style),
                Paragraph(str(f.feature_value), body_style),
                Paragraph(f.contribution_level, bold_style),
                Paragraph(f.direction, body_style),
                Paragraph(f.explanation_text or '', body_style)
            ])

        shap_table = Table(shap_rows, colWidths=[1.5 * inch, 1.0 * inch, 0.9 * inch, 1.1 * inch, 2.7 * inch])
        shap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(shap_table)
        story.append(Spacer(1, 10))

        # Personalized Recommendations
        story.append(Paragraph("4. Personalized Wellness Recommendations", section_heading))
        rec_data = []
        for r in recommendations[:6]:
            rec_data.append([
                Paragraph(f"<b>[{r.category}]</b>", bold_style),
                Paragraph(r.recommendation_text, body_style)
            ])
        if not rec_data:
            rec_data.append([Paragraph("General", bold_style), Paragraph("Maintain regular sleep schedule and hydration.", body_style)])

        rec_table = Table(rec_data, colWidths=[1.8 * inch, 5.4 * inch])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), light_bg),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 14))

        # Disclaimer
        disclaimer_text = (
            "<b>IMPORTANT EDUCATIONAL & WELLNESS DISCLAIMER:</b> "
            "This system is intended for educational, academic research, and wellness awareness purposes and does NOT replace professional medical diagnosis, clinical evaluation, or doctor consultations. "
            "If you suffer from persistent sleep difficulty, fatigue, or breathing irregularities, consult a licensed healthcare professional or sleep physician."
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style))

        doc.build(story)
        buffer.seek(0)
        return buffer
