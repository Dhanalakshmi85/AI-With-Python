# Artificial Intelligence with Python
# Assignment 6: Banking Predictions — Steps 0–10
# Works with UCI Bank Marketing 'bank.csv' (semicolon-delimited).
# Save this as e.g. bank_assignment6.py and run.

# ========== Imports ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ========== Step 0 ==========
# Download "bank.csv" from the UCI/Kaggle page and place it next to this script.
# URL example: https://archive.ics.uci.edu/ml/datasets/bank+marketing

# ========== Step 1 ==========
# Read CSV (note the delimiter=';') and inspect
df = pd.read_csv("bank.csv", delimiter=';')
print("Step 1: Dataset info")
print(df.info())
print(df.head())
print("Shape of dataset:", df.shape)

# ========== Step 2 ==========
# Select required columns
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']].copy()
print("\nStep 2: Selected columns (df2)")
print(df2.head())

# ========== Step 3 ==========
# One-hot encode categoricals; force integer dtypes for clean numerics
df3 = pd.get_dummies(
    df2,
    columns=['job', 'marital', 'default', 'housing', 'poutcome'],
    dtype=int
)

# Map target 'y' to numeric BEFORE any correlations/ML
df3['y'] = df3['y'].map({'yes': 1, 'no': 0}).astype(int)

print("\nStep 3: After get_dummies + y mapping (df3)")
print(df3.head())
print("Shape of df3:", df3.shape)

# ========== Step 4 ==========
# Correlation heatmap (numeric_only avoids non-numeric issues if any creep in)
plt.figure(figsize=(12, 9))
corr = df3.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Step 4: Correlation Heatmap of df3")
plt.tight_layout()
plt.show()

"""
Step 4 - Discussion:
- Dummy variables from the same original category are mutually exclusive, so they often show
  negative correlations with each other.
- Most cross-feature correlations are small in magnitude (weak linear relationships).
"""

# ========== Step 5 ==========
# Target and features
y = df3['y']
X = df3.drop(columns=['y'])
print("\nStep 5: X and y shapes")
print("X:", X.shape, "| y:", y.shape)

# ========== Step 6 ==========
# 75/25 split (stratify to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("\nStep 6: Train/Test split")
print("Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])

# ========== Step 7 & 8: Logistic Regression ==========
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

cm_log = confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)

print("\nStep 7/8: Logistic Regression")
print("Confusion Matrix:\n", cm_log)
print("Accuracy:", acc_log)
print("Classification Report:\n", classification_report(y_test, y_pred_log, digits=3))

plt.figure(figsize=(4, 3))
sns.heatmap(cm_log, annot=True, fmt='d', cbar=False)
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ========== Step 9: K-Nearest Neighbors (k=3) ==========
# KNN benefits from scaling. StandardScaler(with_mean=False) is safe for sparse-like matrices from dummies.
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("\nStep 9: KNN (k=3)")
print("Confusion Matrix:\n", cm_knn)
print("Accuracy:", acc_knn)
print("Classification Report:\n", classification_report(y_test, y_pred_knn, digits=3))

plt.figure(figsize=(4, 3))
sns.heatmap(cm_knn, annot=True, fmt='d', cbar=False)
plt.title("KNN (k=3) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ========== Step 10 ==========
print("\nStep 10: Comparison")
print(f"Logistic Regression Accuracy: {acc_log:.4f}")
print(f"KNN (k=3) Accuracy:           {acc_knn:.4f}")

"""
Step 10 - Summary:
- Logistic Regression is a strong baseline for this binary classification problem and
  often performs well with one-hot encoded features.
- KNN can be competitive but is sensitive to scaling and the choice of k; try k in {3,5,7,9}
  if you want to explore. On larger datasets, KNN can be slower at prediction time.
"""
