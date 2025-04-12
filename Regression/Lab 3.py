import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(0)
x = 6 * np.random.rand(100, 1) - 3
y = 0.5 * x**2 + x + 2  + np.random.randn(100, 1)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly_train = poly_features.fit_transform(x_train)
x_poly_test = poly_features.transform(x_test)

# Create and train the model
model = LinearRegression()
model.fit(x_poly_train, y_train)

# Make predictions
y_pred = model.predict(x_poly_test)

# Print results
print(f"Coefficients:" ,  model.coef_[0])
print(f"Intercept:     {model.intercept_[0]:.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.2f}")

# Plot the results
x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
x_plot_poly = poly_features.transform(x_plot)
y_plot = model.predict(x_plot_poly)

plt.scatter(x_test, y_test, color='blue')
plt.plot(x_plot, y_plot, color='red', linewidth=2)
plt.title('Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()