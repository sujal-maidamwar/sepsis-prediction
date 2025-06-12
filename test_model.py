import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from glob import glob

# Load trained model
model = load_model("models/lstm_model.h5")

# Load test data from .psv files
test_files = glob("test_data/*.psv")

X_test = []
y_test = []

for file in test_files:
    df = pd.read_csv(file, sep='|')
    
    if 'SepsisLabel' not in df.columns:
        continue

    y = int(df['SepsisLabel'].max())
    df.drop('SepsisLabel', axis=1, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)

    if len(df) < 100:
        pad_len = 100 - len(df)
        pad_df = pd.DataFrame(0, index=np.arange(pad_len), columns=df.columns)
        df = pd.concat([df, pad_df], ignore_index=True)
    else:
        df = df.iloc[:100]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    X_test.append(df_scaled)
    y_test.append(y)

X_test = np.array(X_test)
y_test = np.array(y_test)

if len(X_test) == 0:
    print("❌ No valid test data found. Please check your test_data folder and .psv files.")
    exit()

# Make predictions
y_pred_probs = model.predict(X_test).flatten()
y_pred_binary = (y_pred_probs > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_probs)
cm = confusion_matrix(y_test, y_pred_binary)

print("\n📊 Evaluation Metrics:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall (Sensitivity):", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC Score:", round(roc_auc, 4))
print("Confusion Matrix:\n", cm)

# Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Sepsis', 'Sepsis'], yticklabels=['No Sepsis', 'Sepsis'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar chart for scores
metrics = [accuracy, precision, recall, f1, roc_auc]
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

plt.figure(figsize=(8, 5))
colors = sns.color_palette("pastel")
plt.barh(labels, metrics, color=colors)
plt.xlabel("Score")
plt.title("Evaluation Metrics Comparison")
plt.xlim(0, 1)
plt.grid(axis='x')
plt.tight_layout()
plt.show()

