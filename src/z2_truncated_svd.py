"""
Truncated SVD analysis for Latent Semantic Indexing (LSI).
Performs dimensionality reduction on the term-document matrix and generates visualizations.
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
import joblib

# Configuration
ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TDM = PROCESSED / "tdm.npz"
CORPUS = PROCESSED / "corpus.json"
VARIANCE_REPORT = REPORTS / "z2_variance_analysis.txt"
LSI_DOCUMENTS = PROCESSED / "lsi_documents.npy"

KS = [10, 20, 50, 100, 200]
MAX_K = max(KS)


def load_data():
    """Load term-document matrix and corpus."""
    matrix = load_npz(TDM)
    with open(CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)
    return matrix, corpus


def perform_svd(matrix):
    """Perform Truncated SVD on the term-document matrix."""
    svd = TruncatedSVD(n_components=MAX_K, random_state=42)
    documents_lsi = svd.fit_transform(matrix.T)
    return svd, documents_lsi


def save_lsi_documents(documents_lsi):
    """Save LSI document representations to disk."""
    np.save(LSI_DOCUMENTS, documents_lsi)


def create_scree_plot(svd):
    """Generate and save scree plot of singular values."""
    singular_values = svd.singular_values_

    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, len(singular_values) + 1),
        singular_values,
        marker="o",
    )
    plt.xlabel("Component")
    plt.ylabel("Singular value")
    plt.title("Scree Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS / "z2_scree_plot.png")
    plt.close()


def write_variance_report(svd):
    """Write explained variance analysis to report file."""
    explained = svd.explained_variance_ratio_
    cum = np.cumsum(explained)

    with open(VARIANCE_REPORT, "w", encoding="utf-8") as f:
        f.write("Explained Variance Ratio Analysis\n")
        f.write("=" * 40 + "\n\n")

        for k in KS:
            variance_pct = cum[k - 1] * 100
            f.write(f"k={k:3d}: {variance_pct:.2f}%\n")

        f.write("\n" + "=" * 40 + "\n")
        f.write(f"Analysis completed for k values: {KS}\n")


def extract_category(url):
    """Extract category from URL path."""
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    return parts[0] if parts else "root"


def create_lsi_visualization(documents_lsi, corpus):
    """Generate and save LSI space visualization."""
    labels = [extract_category(doc["url"]) for doc in corpus]
    unique = sorted(set(labels))

    mapping = {lab: i for i, lab in enumerate(unique)}
    colors = [mapping[l] for l in labels]

    x = documents_lsi[:, 0]
    y = documents_lsi[:, 1]

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=colors, cmap="tab20", s=8)
    plt.xlabel("LSI 1")
    plt.ylabel("LSI 2")
    plt.title("Documents in LSI space")
    plt.tight_layout()
    plt.savefig(REPORTS / "z2_lsi_documents.png")
    plt.close()


def main():
    """Main execution."""
    print("Loading data...")
    matrix, corpus = load_data()
    print(f"Matrix shape: {matrix.shape}")

    print("Performing Truncated SVD...")
    svd, documents_lsi = perform_svd(matrix)

    print("Saving LSI document matrix...")
    save_lsi_documents(documents_lsi)

    print("Generating scree plot...")
    create_scree_plot(svd)

    print("Writing variance report...")
    write_variance_report(svd)

    print("Creating LSI visualization...")
    create_lsi_visualization(documents_lsi, corpus)

    joblib.dump(svd, PROCESSED / "svd_model.joblib")

    print(f"Report saved to: {VARIANCE_REPORT}")
    print("Done!")


if __name__ == "__main__":
    main()