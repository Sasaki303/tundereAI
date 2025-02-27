from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="This is a test PDF file.", ln=True, align='C')

pdf.output("C:/Users/sasak/tundereAI/local-rag/test.pdf")
