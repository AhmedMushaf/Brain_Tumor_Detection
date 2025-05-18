from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import uuid
from datetime import datetime
import gradio as gr
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import tempfile
import os

# Load model once
model = load_model("brain_tumor_model.h5")
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (150, 150)) / 255.0
    return np.expand_dims(img, axis=0)

def save_report_as_files(report_text, patient_id):
    temp_dir = tempfile.mkdtemp()
    txt_path = os.path.join(temp_dir, f"{patient_id}_report.txt")
    pdf_path = os.path.join(temp_dir, f"{patient_id}_report.pdf")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 40
    for line in report_text.split('\n'):
        if y < 40:
            c.showPage()
            y = height - 40
        try:
            c.drawString(40, y, line)
        except:
            safe_line = line.encode("utf-8", errors="ignore").decode("utf-8")
            c.drawString(40, y, safe_line)
        y -= 15
    c.save()
    return txt_path, pdf_path

def generate_patient_id(name):
    initials = ''.join([x[0].upper() for x in name.split() if x])
    unique_part = uuid.uuid4().hex[:6].upper()
    return f"{initials}-{unique_part}"

def generate_medical_report(name, age, gender, symptoms, image):
    patient_id = generate_patient_id(name)
    img_input = preprocess_image(image)
    pred = model.predict(img_input)
    predicted_index = np.argmax(pred)
    predicted_class = labels[predicted_index]
    tumor_present = "Yes" if predicted_class != "no_tumor" else "No"

    advice = {
        "glioma_tumor": "Gliomas are serious and require immediate consultation with a neuro-oncologist.",
        "meningioma_tumor": "Meningiomas are often benign, but a follow-up MRI and evaluation is recommended.",
        "no_tumor": "No signs of brain tumor detected. Continue routine health check-ups.",
        "pituitary_tumor": "Pituitary tumors may affect hormones. Endocrinologist consultation advised."
    }

    now = datetime.now().strftime('%Y-%m-%d | %H:%M:%S')
    report = f"""
╔══════════════════════════════════════╗
║          MEDICAL DIAGNOSTIC REPORT          ║
╚══════════════════════════════════════╝

🧑‍⚕️ Patient Information
----------------------------------
Name         : {name}
Patient ID   : {patient_id}
Age          : {age}
Gender       : {gender}
Date & Time  : {now}

📝 Clinical Summary
----------------------------------
{symptoms}

🧠 Tumor Detection Results
----------------------------------
Tumor Present : {tumor_present}
Tumor Type    : {predicted_class.replace('_', ' ') if tumor_present == "Yes" else "N/A"}

📌 Medical Advice
----------------------------------
{advice[predicted_class]}

══════════════════════════════════════
This is a computer-generated preliminary report.
Consult a specialist for confirmation and treatment.
══════════════════════════════════════
""".strip()

    txt_path, pdf_path = save_report_as_files(report, patient_id)
    return predicted_class, report, txt_path, pdf_path

iface = gr.Interface(
    fn=generate_medical_report,
    inputs=[
        gr.Textbox(label="Patient Name"),
        gr.Number(label="Age"),
        gr.Radio(choices=["Male", "Female", "Other"], label="Gender"),
        gr.Textbox(label="Symptoms (Clinical Summary)", lines=4),
        gr.Image(type="pil", label="Upload MRI Image")
    ],
    outputs=[
        gr.Textbox(label="Predicted Tumor Type"),
        gr.Textbox(label="Formatted Medical Report"),
        gr.File(label="Download TXT Report"),
        gr.File(label="Download PDF Report")
    ],
    title="🧠 Brain Tumor Classifier & Medical Report Generator",
    description="Upload patient details and MRI image to classify brain tumors and generate a formatted diagnostic report.",
)

iface.launch()
