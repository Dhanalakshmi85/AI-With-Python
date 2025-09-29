import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('weight-height.csv')
X = data["Height"].values.reshape(-1, 1)
y = data["Weight"].values
plt.scatter(X, y, alpha=0.5)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height vs Weight")
plt.show()
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_pred, color="red")  # regression line
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Linear Regression")
plt.show()
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y,y_pred)
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("R2 score:", r2)
print("rmse:",rmse)
