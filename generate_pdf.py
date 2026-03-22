import fitz  # PyMuPDF
import re
import random

# Load dataset
with open('data/mtsamples.txt', encoding='utf-8') as f:
    text = f.read()

# Split into patient records
records = re.split(r'--- NOTE ---', text)

# Create PDF
doc = fitz.open()

for i, record in enumerate(records[:50]):  # limit for testing
    page = doc.new_page()

    # Page layout (margins)
    rect = fitz.Rect(50, 80, 550, 800)

    # Clean text
    record = record.strip()

    # Add spacing for readability
    record = re.sub(
        r'(CC:|HX:|EXAM:|IMPRESSION:|PROCEDURE:)',
        r'\n\n\1\n',
        record
    )

    # Add fake hospital header (realistic testing)
    hospital_name = "ABC Hospital - Patient Record"
    patient_id = f"PID-{random.randint(1000,9999)}"

    page.insert_text((50, 30), hospital_name, fontsize=12)
    page.insert_text((400, 30), patient_id, fontsize=10)

    # Insert structured text
    page.insert_textbox(
        rect,
        record,
        fontsize=10,
        fontname="helv",
        align=0  # left align
    )

# Save PDF
doc.save("patient_records_realistic.pdf")
doc.close()

print("✅ PDF created: patient_records_realistic.pdf")