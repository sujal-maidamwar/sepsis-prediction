Sepsis Risk Prediction Using ICU Patient Data

This project aims to detect sepsis early in ICU patients using a deep learning model (LSTM) trained on real-world time-series ICU data. The system provides early risk scores via a simple web interface, helping medical staff act promptly.

Project Overview

Goal: Predict early signs of sepsis using ICU vitals and lab results.

Model Used: LSTM (Long Short-Term Memory) neural network.

Frameworks: Python, TensorFlow/Keras, Flask, Pandas, scikit-learn

Frontend: Flask + HTML/CSS

Dataset: PhysioNet 2019 Sepsis Challenge dataset (over 40,000 ICU records)

Features

Upload .psv ICU patient data

Real-time prediction of sepsis probability

Color-coded risk indicator (Low / Moderate / High)

Risk trendline graph

Summary of input vitals

Plan to add SHAP-based explainability (future scope)

Folder Structure

sepsis_prdiction_project/
├── models/                  # Contains trained LSTM model (lstm_model.h5)
├── static/                  # HTML/CSS/JS assets for Flask
├── templates/               # Dashboard HTML templates
├── dataset_A/               # (Optional) Training data folder
├── app.py                   # Main Flask web application
├── train_model.py           # Script to train LSTM model
├── test_model.py            # Evaluation script (Accuracy, ROC, etc.)
├── README.md                # Project documentation

How to Run

1. Install Dependencies

pip install -r requirements.txt

Or manually:

pip install tensorflow pandas numpy matplotlib seaborn flask scikit-learn

2. Train the Model (Optional)

python train_model.py

Only required if you want to retrain using your own data.

3. Run the Flask App

python app.py

Open your browser at: http://localhost:5000

Technologies Used
 GitHub Repository Access & Update Instructions

GitHub Repo URL: https://github.com/sujal-maidamwar/sepsis-prediction.git

🔄 Clone the Repository:

git clone https://github.com/sujal-maidamwar/sepsis-prediction.git
cd sepsis-prediction

📝 Make Changes:

Edit files as needed in VS Code or any editor.

📤 Commit and Push Changes:

git add .
git commit -m "Updated project files"
git push origin main

🔒 Make sure you're logged into your GitHub account and have write access.


LSTM: For sequence modeling over time-series vitals

Flask: Lightweight backend to serve predictions

StandardScaler: Normalize feature values

Matplotlib / Seaborn: For risk graph and SHAP visualization

Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

ROC-AUC Score

Confusion Matrix

You can run test_model.py to evaluate the model using test .psv files.

Future Scope

Add SHAP explainability

Accept .csv and .xlsx file formats

Hybrid models (LSTM + GRU / Transformer)

Mobile/web app version

Alert system integration (SMS, Email)

Cloud deployment for hospital integration

Team & Guide

Guide: Prof. Anuradha Hiwase

Team Members: Sujal, Akanksha, Kalyani, Reeya, Yash, Parth

License

MIT License – Free to use and modify for educational and research purposes.

