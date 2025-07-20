# 🦠 Pneumonia Detection using Deep Learning

This project uses **Convolutional Neural Networks (CNNs)** and **LSTM models** to detect pneumonia from chest X-ray images.  
The web application allows users to upload images for real-time diagnosis through a **Flask-based interface**.

---

## 🧠 Project Overview

This repository includes:

- Pre-trained models: `PNEUMONIA.h5`, `bestmodel.h5`, `lstmchar256256128test.h5`
- A Flask web app: `app.py` to serve predictions
- Jupyter Notebook: `t.ipynb` for model training, testing, and evaluation
- Supporting datasets and download links (`Data link.txt`)
- Front-end templates (`templates/`) and upload folder (`uploads/`)

---

## 📦 Model Details

- `PNEUMONIA.h5`: CNN model trained to classify chest X-ray images as **Normal** or **Pneumonia**.
- `bestmodel.h5`: Optimized version of the CNN model after tuning and validation.
- `lstmchar256256128test.h5`: LSTM model (potentially used for text or sequential medical data).

---

## 📊 Dataset

- Chest X-ray dataset links are provided in the `Data link.txt` file.
- Make sure the datasets are properly structured before training or testing.

---

## 📓 Notebooks

- `t.ipynb`: Contains training, evaluation, and experimentation code for the models.

---

## 💻 How to Run

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
