from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import uuid

# Import custom utility functions
from utils import preprocess_patient_data, generate_summary, get_interpretation

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'

# Create folders if not present
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the trained model once
MODEL_PATH = 'models/lstm_model.h5'
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded.')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            # Preprocess data and get prediction info
            processed_data, trend, risk_percent, risk_label, scaler = preprocess_patient_data(file_path, model)

            # Generate summary info from raw file
            summary = generate_summary(file_path)

            interpretation = get_interpretation(risk_label)

            # Plot and save risk trend graph
            plot_id = str(uuid.uuid4())[:8]
            plot_path = os.path.join(RESULT_FOLDER, f"risk_trend_{plot_id}.png")

            plt.figure(figsize=(6, 3))
            plt.plot(trend, marker='o', linestyle='-', color='b')
            plt.title("Sepsis Risk Trend")
            plt.xlabel("Hour")
            plt.ylabel("Risk Score")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            return render_template('index.html',
                                   risk_percent=round(risk_percent, 2),
                                   risk_label=risk_label,
                                   trend_image=plot_path,
                                   summary=summary,
                                   interpretation=interpretation)

        except Exception as e:
            return render_template('index.html', error=f"Error processing file: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

