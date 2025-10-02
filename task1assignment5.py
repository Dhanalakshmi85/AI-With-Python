import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
diabetes = load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target
print("=== Diabetes Progression Prediction ===")
X_base = X[['bmi', 's5']]
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Model: BMI + S5")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", rmse)
X_bp = X[['bmi', 's5', 'bp']]
X_train, X_test, y_train, y_test = train_test_split(X_bp, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nModel: BMI + S5 + BP")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", rmse)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nModel: All variables")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", rmse)
