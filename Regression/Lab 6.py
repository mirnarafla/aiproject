import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Example dataset (replace with your actual data)
X = np.random.rand(100, 1)  # Features
y = 3 * X.squeeze() + np.random.randn(100) * 0.5  # Target with some noise

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement OLS manually
x_with_intercept = np.c_[np.ones((x_train.shape[0], 1)), x_train]
beta_hat = np.linalg.inv(x_with_intercept.T.dot(x_with_intercept)).dot(x_with_intercept.T).dot(y_train)

print("OLS Coefficients (including intercept):", beta_hat.flatten())

# Compare with sklearn's LinearRegression
sklearn_model = LinearRegression(fit_intercept=True)
sklearn_model.fit(x_train, y_train)

print("\nScikit-learn Coefficients:", sklearn_model.coef_.flatten())
print(f"Scikit-learn Intercept: {sklearn_model.intercept_:.2f}")