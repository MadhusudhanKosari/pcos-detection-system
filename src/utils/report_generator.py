from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_pdf(data):
    path = "static/reports/pcos_report.pdf"
    c = canvas.Canvas(path, pagesize=A4)

    text = c.beginText(40, 800)
    for line in data:
        text.textLine(line)

    c.drawText(text)
    c.save()

    return path
