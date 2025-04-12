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

# Create and train Lasso model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)

# Make predictions
y_pred_lasso = lasso_model.predict(x_test)

# Print results
print(f"Lasso Coefficients:" ,  lasso_model.coef_)
print(f"Lasso Intercept:     {lasso_model.intercept_}")
print(f"Lasso R-squared: {r2_score(y_test, y_pred_lasso):.2f}")