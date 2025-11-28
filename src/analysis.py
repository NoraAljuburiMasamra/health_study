
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

class HealthAnalyzer:
    def __init__(self, data):
        self.data = data

    def compute_confidence_interval(self, bootstrap_diffs):
        ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])
        return ci_low, ci_high

    def plot_distribution(self, bootstrap_diffs):
        plt.hist(bootstrap_diffs, bins=30, color='skyblue', edgecolor='black')
        plt.title("Bootstrap Distribution")
        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.show()

    # Linjär regression
    def linear_regression(self, features, target):
        X = self.data[features]
        y = self.data[target]
        model = LinearRegression()
        model.fit(X, y)
        return model

    # PCA
    def perform_pca(self, features, n_components=2):
        X = self.data[features]
        X_std = (X - X.mean()) / X.std()
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_std)
        return pca, X_pca

    def plot_pca(self, X_pca):
        plt.scatter(X_pca[:, 0], X_pca[:, 1], color='green')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA på valda variabler')
        plt.show()
