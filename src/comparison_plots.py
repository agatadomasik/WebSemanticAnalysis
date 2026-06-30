from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_agreement_plot(results: list[dict], path: Path) -> None:
    ks = [r["k"] for r in results]
    nmis = [r["nmi"] for r in results]
    aris = [r["ari"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, nmis, marker="o", label="NMI")
    plt.plot(ks, aris, marker="o", label="ARI")
    plt.xlabel("k (semantic clusters)")
    plt.ylabel("Agreement score")
    plt.title("Semantic vs structural agreement across k")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_confusion_heatmap(matrix: np.ndarray, top_communities: int, path: Path) -> None:
    """Trim to the N largest communities (by row sum) so the heatmap stays readable."""
    row_sums = matrix.sum(axis=1)
    top_idx = np.argsort(row_sums)[::-1][:top_communities]
    trimmed = matrix[top_idx]

    plt.figure(figsize=(10, 8))
    plt.imshow(trimmed, aspect="auto", cmap="viridis")
    plt.colorbar(label="number of documents")
    plt.xlabel("Semantic cluster (k-means)")
    plt.ylabel(f"Community (top {top_communities} by size)")
    plt.yticks(range(len(top_idx)), [f"C{i}" for i in top_idx])
    plt.title("Communities x semantic clusters")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
