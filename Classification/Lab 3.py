import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Generate a sample dataset for Classification
x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(x_train_scaled, y_train)

# Make predictions
y_pred_dt = dt_model.predict(x_test_scaled)

# Evaluate the model
print("Decision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=[f'Feature {i}' for i in range(20)], class_names=['0', '1'])
plt.title('Decision Tree Visualization')
plt.show()

# Plot feature importance
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.title('Feature Importances')
plt.bar(range(20), importances[indices])
plt.xticks(range(20), [f'Feature {i}' for i in indices], rotation=90)
plt.tight_layout()
plt.show()