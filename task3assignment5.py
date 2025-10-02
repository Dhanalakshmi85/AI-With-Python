import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

"""
We predict 'mpg' using multiple regression.
We exclude 'mpg', 'name', and 'origin'.
We apply Ridge and Lasso with different alpha values,
plot R2 scores, and find the best alpha.
"""

# Load dataset
auto = pd.read_csv("Auto.csv")

print("=== Car MPG Prediction ===")
print("Variables:", auto.columns.tolist())

# Drop unused columns
X = auto.drop(columns=['mpg', 'name', 'origin'])
y = auto['mpg']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge and Lasso
alphas = np.logspace(-3, 2, 20)
ridge_scores, lasso_scores = [], []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(ridge.score(X_test, y_test))

    lasso = Lasso(alpha=alpha, max_iter=5000)
    lasso.fit(X_train, y_train)
    lasso_scores.append(lasso.score(X_test, y_test))

# Plot results
plt.plot(alphas, ridge_scores, label="Ridge R2")
plt.plot(alphas, lasso_scores, label="Lasso R2")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("R2 Score")
plt.title("R2 Score vs Alpha (Ridge & Lasso)")
plt.legend()
plt.show()

# Best alpha values
best_ridge_alpha = alphas[np.argmax(ridge_scores)]
best_lasso_alpha = alphas[np.argmax(lasso_scores)]

print("\nBest Ridge alpha:", best_ridge_alpha, "R2:", max(ridge_scores))
print("Best Lasso alpha:", best_lasso_alpha, "R2:", max(lasso_scores))
