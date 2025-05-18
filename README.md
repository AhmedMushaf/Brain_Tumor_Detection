# 🧠 Brain Tumor Detection & Medical Report Generator

A deep learning-based web application that classifies brain tumors from MRI scans using a VGG16 CNN model and generates downloadable medical reports (TXT & PDF) with patient details, diagnosis, and expert advice.

---

## 📁 Project Structure

- `train_model.py`: Trains and saves the VGG16-based classification model and performance graphs.
- `app_predict.py`: Gradio-based frontend for predictions and automatic medical report generation.
- `brain_tumor_model.h5`: Trained Keras model.
- `labels.pkl`: Encoded list of tumor labels used for decoding predictions.
- `accuracy_plot.png` & `loss_plot.png`: Training history graphs.

---

## 🧬 Dataset

MRI brain scan images categorized into:
- `glioma_tumor`
- `meningioma_tumor`
- `no_tumor`
- `pituitary_tumor`

📂 Folder structure:
dataset/
├── Training/
│ ├── glioma_tumor/
│ ├── meningioma_tumor/
│ ├── no_tumor/
│ └── pituitary_tumor/
├── Testing/
│ └── (same as above)
