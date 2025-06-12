import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_patient_data(filepath, model, max_timesteps=100):
    # Load the .psv file
    df = pd.read_csv(filepath, sep='|')

    # Remove the label if present
    if 'SepsisLabel' in df.columns:
        df.drop(['SepsisLabel'], axis=1, inplace=True)

    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)

    # Pad or crop to fixed length
    if len(df) < max_timesteps:
        pad_len = max_timesteps - len(df)
        pad_df = pd.DataFrame(0, index=np.arange(pad_len), columns=df.columns)
        df = pd.concat([df, pad_df], ignore_index=True)
    else:
        df = df.iloc[:max_timesteps]

    # Scale the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    input_data = np.expand_dims(df_scaled, axis=0)  # shape: (1, timesteps, features)

    # Predict risk score
    risk_score = model.predict(input_data)[0][0] * 100  # Convert to percentage

    # Predict trend across timesteps (optional simulation)
    trend = [float(model.predict(np.expand_dims(df_scaled[:i+1], axis=0))[0][0]) * 100 for i in range(len(df_scaled))]

    # Risk label
    if risk_score < 30:
        label = 'Low'
    elif risk_score < 70:
        label = 'Moderate'
    else:
        label = 'High'

    return df_scaled, trend, risk_score, label, scaler

def generate_summary(filepath):
    df = pd.read_csv(filepath, sep='|')
    summary = {}

    # Sample stats to show
    summary['Age (if present)'] = int(df['Age'].iloc[0]) if 'Age' in df.columns else 'Unknown'
    summary['Avg HR'] = round(df['HR'].mean(), 2) if 'HR' in df.columns else 'N/A'
    summary['Max WBC'] = round(df['WBC'].max(), 2) if 'WBC' in df.columns else 'N/A'
    summary['Min MAP'] = round(df['MAP'].min(), 2) if 'MAP' in df.columns else 'N/A'
    summary['Duration (hours)'] = len(df)

    return summary

def get_interpretation(risk_level):
    if risk_level == 'Low':
        return "Patient is stable. Continue monitoring vitals regularly."
    elif risk_level == 'Moderate':
        return "Patient is showing early signs. Increase monitoring and review lab trends."
    else:
        return "High risk of sepsis detected. Immediate clinical attention recommended."
