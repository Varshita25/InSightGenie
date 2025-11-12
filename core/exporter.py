# core/exporter.py
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches, Pt
import tempfile
import os
import re

class UTF8PDF(FPDF):
    def __init__(self):
        super().__init__()
        try:
            # Try to use Arial Unicode MS if available (better Unicode support)
            self.add_font('Arial Unicode MS', '', 'arial-unicode-ms.ttf', uni=True)
            self.default_font = 'Arial Unicode MS'
        except Exception:
            # Fallback to Arial
            self.default_font = 'Arial'

    def sanitize_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        # Replace problematic characters with similar ASCII ones
        text = text.encode('ascii', errors='replace').decode()
        # Remove any remaining non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\r', '\t'])
        return text

    def safe_cell(self, w, h, txt='', border=0, ln=0, align='', fill=False):
        self.cell(w, h, self.sanitize_text(txt), border, ln, align, fill)

    def safe_multi_cell(self, w, h, txt='', border=0, align='', fill=False):
        self.multi_cell(w, h, self.sanitize_text(txt), border, align, fill)

def build_pdf(report_data: dict, title="Data Insights Report"):
    try:
        # Try UTF8 PDF first
        pdf = UTF8PDF()
    except Exception as e:
        print(f"Warning: Using standard PDF due to: {str(e)}")
        pdf = FPDF()
    
    def safe_write(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Warning: Error in PDF generation: {str(e)}")
            # Try to sanitize all string inputs
            args = tuple(str(arg).encode('ascii', errors='replace').decode() if isinstance(arg, str) else arg for arg in args)
            return func(*args, **kwargs)
    
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    try:
        pdf.set_font("DejaVu", "B", 24)
    except Exception:
        pdf.set_font("Arial", "B", 24)
    pdf.safe_cell(0, 20, title, ln=True, align="C")
    pdf.ln(10)
    
    # Dataset overview
    ov = report_data.get("overview", {})
    try:
        pdf.set_font("DejaVu", "B", 16)
    except Exception:
        pdf.set_font("Arial", "B", 16)
    pdf.set_fill_color(240, 240, 240)
    pdf.safe_cell(0, 10, "Dataset Overview", ln=True, fill=True)
    pdf.ln(5)
    try:
        pdf.set_font("DejaVu", "", 12)
    except Exception:
        pdf.set_font("Arial", size=12)
    pdf.safe_multi_cell(0, 8, txt=ov.get("summary", ""))
    pdf.ln(8)
    
    # Add dataset info if available
    if "dataset_name" in ov:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Dataset: {ov['dataset_name']}", ln=True)
    if "shape" in ov:
        pdf.cell(0, 8, f"Size: {ov['shape']}", ln=True)
    pdf.ln(5)

    # EDA
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Exploratory Data Analysis", ln=True)
    pdf.set_font("Arial", size=11)
    for k, v in report_data.get("eda", {}).items():
        pdf.multi_cell(0, 8, f"{k.title()}: " + "; ".join(v))
    pdf.ln(4)

    # Hypotheses
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Hypothesis Testing", ln=True)
    pdf.set_font("Arial", size=11)
    for h in report_data.get("hypotheses", []):
        pdf.multi_cell(0, 8, f"- {h['title']} ({h['test']}): {h['result']} → {h['interpretation']}")
    pdf.ln(4)

    # Suggestions
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Suggested Analyses", ln=True)
    pdf.set_font("Arial", size=11)
    for s in report_data.get("suggestions", []):
        pdf.multi_cell(0, 8, f"- {s}")
    pdf.ln(4)

    # Q&A
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Ask-the-Data Q&A", ln=True)
    pdf.set_font("Arial", size=11)
    for qa in report_data.get("qa", []):
        pdf.multi_cell(0, 8, f"Q: {qa['q']}\nA: {qa['a']}\n")

    # Save
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name


def build_ppt(report_data: dict, title="Data Insights Report"):
    prs = Presentation()

    # Title Slide
    slide_layout = prs.slide_layouts[0]  # Title
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = title
    subtitle = slide.placeholders[1]
    
    # Add dataset info to title slide
    ov = report_data.get("overview", {})
    subtitle_text = []
    if "dataset_name" in ov:
        subtitle_text.append(f"Dataset: {ov['dataset_name']}")
    if "shape" in ov:
        subtitle_text.append(f"Size: {ov['shape']}")
    if subtitle_text:
        subtitle.text = "\n".join(subtitle_text)

    # Overview Slide
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Dataset Overview"
    body = slide.placeholders[1]
    overview_text = []
    if "summary" in ov:
        overview_text.append(ov["summary"])
    if "dtypes" in ov:
        overview_text.append("\nData Types:")
        for col, dtype in ov["dtypes"].items():
            overview_text.append(f"• {col}: {dtype}")
    body.text = "\n".join(overview_text)

    # EDA Slide
    eda = report_data.get("eda", {})
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Exploratory Data Analysis"
    body = slide.placeholders[1]
    lines = []
    for k, v in eda.items():
        lines.append(f"{k.title()}: " + "; ".join(v))
    body.text = "\n".join(lines)

    # Hypotheses Slide
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Hypothesis Testing"
    body = slide.placeholders[1]
    body.text = "\n".join([f"{h['title']} ({h['test']}): {h['result']}" for h in report_data.get("hypotheses", [])])

    # Suggestions Slide
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Suggested Analyses"
    body = slide.placeholders[1]
    body.text = "\n".join(report_data.get("suggestions", []))

    # Q&A Slides (one question per slide for better readability)
    for qa in report_data.get("qa", []):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Q&A Insights"
        body = slide.placeholders[1]
        text = f"Question:\n{qa.get('q', '')}\n\nAnswer:\n{qa.get('a', '')}"
        body.text = text
        
    # Save the presentation
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(tmp.name)
    return tmp.name
    body = slide.placeholders[1]
    body.text = "\n".join([f"Q: {qa['q']}\nA: {qa['a']}" for qa in report_data.get("qa", [])])

    # Figures (if any)
    for fig in report_data.get("figures", []):
        if os.path.exists(fig):
            slide = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
            slide.shapes.title.text = "Visualization"
            left, top, width, height = Inches(1), Inches(1.5), Inches(7.5), Inches(4.5)
            slide.shapes.add_picture(fig, left, top, width, height)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(tmp.name)
    return tmp.name
