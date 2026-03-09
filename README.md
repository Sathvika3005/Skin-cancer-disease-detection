# 🧠 Skin Disease Detection using Deep Learning

A deep learning-based web application that detects different types of skin diseases from images using a Convolutional Neural Network (CNN) with MobileNetV2 . The system also provides visual explanations using Grad-CAM to highlight the region of the image that influenced the prediction.

---

## 📌 Project Overview

Skin diseases are among the most common health problems worldwide. Early detection can help in timely treatment.
This project uses a deep learning model trained on dermoscopic images to classify skin lesions into different disease categories.

Users can upload an image of a skin lesion, and the system will:

* Predict the type of skin disease
* Display the confidence score
* Generate a Grad-CAM heatmap to show the region used by the model for prediction
* Reject non-skin images using a confidence threshold

---

## 🚀 Features

* Image upload interface
* Deep Learning CNN model for classification
* Prediction confidence score
* Grad-CAM visualization for explainable AI
* Rejection system for non-skin images
* Interactive web application interface

---

## 🧪 Dataset

The model is trained on the **HAM10000 Skin Lesion Dataset**, which contains dermoscopic images of different types of skin lesions.

Classes included in the dataset:

1. Actinic Keratoses
2. Basal Cell Carcinoma
3. Benign Keratosis
4. Dermatofibroma
5. Melanoma
6. Nevus
7. Vascular Lesion

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* Grad-CAM (Explainable AI)

---

## 📂 Project Structure

```
skin-disease-detection
│
├── app.py
├── skin_cancer_model.keras
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/yourusername/skin-disease-detection.git
```

Navigate to the project folder:

```
cd skin-disease-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

---

## 🌐 Web Application

The application allows users to upload a skin image and get a prediction along with a Grad-CAM visualization that highlights the important regions of the image.

---


## 👩‍💻 Author

Developed by K.Sathvika
B.Tech Computer Science Student (AI & DS)

---
 
