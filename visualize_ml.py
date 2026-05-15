import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

def plot_ml_clusters(n_samples=300, centers=4):
    """
    Example showing qualitative vs sequential cmaps in ML.
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Qualitative: Good for identifying distinct clusters
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.7)
    ax1.set_title("Qualitative (tab10): Correct for Clusters")
    plt.colorbar(scatter1, ax=ax1)

    # 2. Sequential: Often used for confidence or distance
    # Here we use the distance from the origin as a continuous value
    dist = np.linalg.norm(X, axis=1)
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=dist, cmap='viridis', alpha=0.7)
    ax2.set_title("Sequential (viridis): Good for Continuous Metrics")
    plt.colorbar(scatter2, ax=ax2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Best Practice Tip: 
    # Use 'viridis' or 'magma' for sequential data as they are perceptually uniform 
    # and printer-friendly. Use 'RdBu' or 'coolwarm' for error analysis.
    plot_ml_clusters()
