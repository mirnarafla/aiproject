# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

digits = load_digits()
X_digits = digits.data

pca = PCA(0.95)  # Keep 95% of variance
X_projected = pca.fit_transform(X_digits)
X_reconstructed = pca.inverse_transform(X_projected)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(5):
    axes[0, i].imshow(X_digits[i].reshape(8, 8), cmap='gray')
    axes[0, i].set_title(f'Original')
    axes[1, i].imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
    axes[1, i].set_title(f'Reconstructed')

plt.tight_layout()
plt.show()

print(f"Original dimensions: {X_digits.shape[1]}")
print(f"Reduced dimensions: {X_projected.shape[1]}")