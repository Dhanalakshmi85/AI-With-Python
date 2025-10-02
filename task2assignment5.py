import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
startups = pd.read_csv("50_Startups.csv")
print("=== Profit Prediction ===")
print("Columns:", startups.columns.tolist())
print("\nCorrelation matrix:")
print(startups.corr(numeric_only=True))
sns.heatmap(startups.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - 50_Startups")
plt.show()
X = startups[['R&D Spend', 'Administration', 'Marketing Spend']]
y = startups['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
print("\nTraining RMSE:", train_rmse)
print("Training R2:", r2_score(y_train, y_train_pred))
print("Testing RMSE:", test_rmse)
print("Testing R2:", r2_score(y_test, y_test_pred))
