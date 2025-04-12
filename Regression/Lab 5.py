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

# Create and train Ridge model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)

# Make predictions
y_pred_ridge = ridge_model.predict(x_test)

# Print results
print("Ridge Coefficients:" ,  ridge_model.coef_)
print(f"Ridge Intercept:     {ridge_model.intercept_}")
print(f"Ridge R-squared: {r2_score(y_test, y_pred_ridge):.2f}")