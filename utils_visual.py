import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_pca_2d(X, y, class_names=None, title="PCA - 2D Projection"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    if class_names is not None:
        legend_labels = np.unique(y)
        plt.legend(handles=scatter.legend_elements()[0], labels=[class_names[i] for i in legend_labels])
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.show()


def plot_pca_3d(X, y, class_names=None, title="PCA - 3D Projection"):

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', edgecolor='k', s=50)

    if class_names is not None:
        legend_labels = np.unique(y)
        ax.legend(handles=scatter.legend_elements()[0], labels=[class_names[i] for i in legend_labels])
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    plt.show()