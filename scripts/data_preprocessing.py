import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_folder):
    all_files = [f for f in os.listdir(data_folder) if f.endswith(".psv")]
    data_list = []
    labels = []

    for file in all_files:
        df = pd.read_csv(os.path.join(data_folder, file), sep='|')
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)

        # Drop non-useful columns
        if 'SepsisLabel' not in df.columns:
            continue
        y = int(df['SepsisLabel'].max())  # 1 if patient ever became septic
        df.drop(['SepsisLabel'], axis=1, inplace=True)

        data_list.append(df)
        labels.append(y)

    # Pad sequences to same length
    max_len = max([len(x) for x in data_list])
    for i in range(len(data_list)):
        data_list[i] = data_list[i].reindex(range(max_len), fill_value=0)

    # Convert to 3D array [samples, timesteps, features]
    X = np.stack([df.values for df in data_list])
    y = np.array(labels)

    # Normalize
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

    return X_scaled, y
