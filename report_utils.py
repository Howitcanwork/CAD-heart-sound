"""
report_utils.py
================
Generates a downloadable PDF diagnosis report using fpdf2 (pure-Python,
no system dependencies - safe for Streamlit Community Cloud).
"""

from datetime import datetime
from fpdf import FPDF

from config import (
    PROJECT_TITLE, CLASS_FULL_NAMES, CLINICAL_RECOMMENDATIONS, DISCLAIMER,
    DEPARTMENT, UNIVERSITY, MODEL_NAME, MODEL_ACCURACY,
)


class DiagnosisReportPDF(FPDF):
    def header(self):
        self.set_fill_color(11, 94, 215)
        self.rect(0, 0, 210, 22, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.set_xy(10, 6)
        self.cell(0, 10, "CardioSound AI - Diagnosis Report", ln=False)
        self.set_font("Helvetica", "", 9)
        self.set_xy(10, 14)
        self.cell(0, 6, PROJECT_TITLE[:90], ln=False)
        self.set_text_color(0, 0, 0)
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}  |  {DEPARTMENT}, {UNIVERSITY}", align="C")


def generate_pdf_report(
    predicted_class: str,
    confidence: float,
    probs_dict: dict,
    signal_info: dict,
    true_label: str = None,
) -> bytes:
    pdf = DiagnosisReportPDF()
    pdf.add_page()
    pdf.set_margins(10, 25, 10)
    pdf.set_auto_page_break(auto=True, margin=18)

    # ---- Meta ----
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 7, f"Model: {MODEL_NAME}  (Test Accuracy: {MODEL_ACCURACY})", ln=True)
    pdf.ln(3)

    # ---- Diagnosis summary ----
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(11, 94, 215)
    pdf.cell(0, 9, "AI Diagnosis Summary", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    full_name = CLASS_FULL_NAMES.get(predicted_class, predicted_class)
    pdf.cell(0, 8, f"Predicted Condition: {predicted_class} - {full_name}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Model Confidence: {confidence * 100:.1f}%", ln=True)
    if true_label:
        match = "MATCH" if true_label == predicted_class else "MISMATCH"
        pdf.cell(0, 7, f"Reference / Ground-truth label: {true_label}  ({match})", ln=True)
    pdf.ln(2)

    # ---- Probability table ----
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Class Probability Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for cls, p in probs_dict.items():
        pdf.cell(60, 6, f"{cls} ({CLASS_FULL_NAMES.get(cls, cls)})", border=0)
        pdf.cell(0, 6, f"{p * 100:.2f}%", ln=True)
    pdf.ln(2)

    # ---- Signal info ----
    if signal_info:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Recording Information", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Duration: {signal_info.get('duration_sec', '-')} s", ln=True)
        pdf.cell(0, 6, f"Sampling Rate: {signal_info.get('sampling_rate', '-')} Hz", ln=True)
        pdf.cell(0, 6, f"Signal Quality: {signal_info.get('quality', '-')}", ln=True)
        pdf.ln(2)

    # ---- Clinical recommendation ----
    rec = CLINICAL_RECOMMENDATIONS.get(predicted_class)
    if rec:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Clinical Recommendation", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, rec["summary"], new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for action in rec["actions"]:
            # EDIT 6: return to the left margin on the next line after each
            # action, otherwise the following one starts at the right page
            # edge and gets cut off.
            pdf.multi_cell(0, 6, f"- {action}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"Suggested urgency: {rec['urgency']}", ln=True)
        pdf.ln(2)

    # ---- Disclaimer ----
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 90, 0)
    pdf.multi_cell(0, 6, f"Disclaimer: {DISCLAIMER}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    return bytes(pdf.output(dest="S"))
