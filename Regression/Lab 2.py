import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(0)
x = np.random.rand(100, 3)
y = 4  + np.dot(x, np.array([2, 3, 4])) + np.random.randn(100)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Print results
print(f"Coefficients:" ,  model.coef_)
print(f"Intercept:    {model.intercept_:.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.2f}")

