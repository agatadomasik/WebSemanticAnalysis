"""Cluster documents in the LSI space and summarize the resulting groups."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from src.config import KMEANS_KS, KMEANS_RANDOM_STATE

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

LSI = PROCESSED / "lsi_documents.npy"
VOCAB = PROCESSED / "vocab.json"
CORPUS = PROCESSED / "corpus.json"
SVD_MODEL = PROCESSED / "svd_model.joblib"

KS = KMEANS_KS
RANDOM_STATE = KMEANS_RANDOM_STATE


def load_inputs() -> tuple[np.ndarray, list[dict], dict[str, int], object]:
    """Load LSI embeddings, corpus metadata, vocabulary, and the SVD model."""
    documents = np.load(LSI)
    documents = normalize(documents, norm="l2")

    with open(CORPUS, encoding="utf-8") as handle:
        corpus = json.load(handle)

    with open(VOCAB, encoding="utf-8") as handle:
        vocab = json.load(handle)

    svd = joblib.load(SVD_MODEL)
    return documents, corpus, vocab, svd


def model_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cluster rows of X using k-means and return labels, centroids, and inertia."""
    model = KMeans(
        n_clusters=k,
        n_init=10,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )
    labels = model.fit_predict(X)
    return labels, model.cluster_centers_, float(model.inertia_)

def my_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
):
    """
    Własna implementacja k-means.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Znormalizowane wektory dokumentów (LSI).

    k : int
        Liczba klastrów.

    max_iter : int
        Maksymalna liczba iteracji.

    Returns
    -------
    labels : ndarray (n_samples,)
        Numer klastra każdego dokumentu.

    centroids : ndarray (k, n_features)
        Centroidy.

    inertia : float
        WCSS (Within Cluster Sum of Squares).
    """
    tol = 1e-4

    rng = np.random.default_rng(42)

    n_samples = X.shape[0]

    # Losowy wybór początkowych centroidów
    indices = rng.choice(n_samples, size=k, replace=False)
    centroids = X[indices].copy()

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):

        old_centroids = centroids.copy()

        #
        # Assignment step
        #

        # Odległość euklidesowa od każdego centroidu
        distances = np.linalg.norm(
            X[:, np.newaxis] - centroids[np.newaxis, :],
            axis=2
        )

        labels = np.argmin(distances, axis=1)

        #
        # Update step
        #

        for i in range(k):

            cluster_points = X[labels == i]

            # pusty klaster -> losowa reinicjalizacja
            if len(cluster_points) == 0:
                centroids[i] = X[rng.integers(n_samples)]
                continue

            centroids[i] = cluster_points.mean(axis=0)

        #
        # Kryterium stopu
        #

        shift = np.linalg.norm(centroids - old_centroids)

        if shift < tol:
            break

    #
    # WCSS
    #

    inertia = 0.0

    for i in range(k):

        cluster_points = X[labels == i]

        if len(cluster_points) == 0:
            continue

        inertia += np.sum(
            (cluster_points - centroids[i]) ** 2
        )

    return labels, centroids, inertia

def evaluate_clusters(documents: np.ndarray) -> list[dict]:
    """Run clustering for each requested k and collect quality metrics."""
    results: list[dict] = []

    for k in KS:
        labels, centroids, inertia = my_kmeans(documents, k)
        # labels, centroids, inertia = model_kmeans(documents, k)
        silhouette = silhouette_score(documents, labels, metric="cosine")

        result = {
            "k": k,
            "labels": labels,
            "centroids": centroids,
            "wcss": inertia,
            "silhouette": silhouette,
        }
        results.append(result)

        print(
            f"k={k:2d}"
            f"   WCSS={inertia:.2f}"
            f"   silhouette={silhouette:.3f}"
        )

    return results


def save_plots(results: list[dict]) -> None:
    """Save elbow and silhouette plots to the reports directory."""
    ks = [result["k"] for result in results]
    wcss = [result["wcss"] for result in results]
    silhouettes = [result["silhouette"] for result in results]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, wcss, marker="o")
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.title("Elbow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS / "z3_elbow.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ks, silhouettes, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS / "z3_silhouette.png")
    plt.close()


def describe_clusters(
    centroids: np.ndarray,
    vocab: dict[str, int],
    svd: object,
    top_n: int = 10,
) -> list[list[tuple[str, float]]]:
    """Return the most characteristic terms for each cluster centroid."""
    idx_to_term = {idx: term for term, idx in vocab.items()}
    descriptions: list[list[tuple[str, float]]] = []

    for centroid in centroids:
        term_weights = centroid @ svd.components_
        top_idx = np.argsort(term_weights)[::-1][:top_n]

        description = [(idx_to_term[i], float(term_weights[i])) for i in top_idx]
        descriptions.append(description)

    return descriptions


def print_cluster_examples(best_result: dict, corpus: list[dict], report_handle) -> None:
    """Write sample URLs for each cluster into the report."""
    for cluster in range(best_result["centroids"].shape[0]):
        report_handle.write("\n")
        report_handle.write("=" * 60 + "\n")
        report_handle.write(f"CLUSTER {cluster}\n")

        cluster_indices = np.where(best_result["labels"] == cluster)[0]
        for index in cluster_indices[:3]:
            report_handle.write(f"{corpus[index]['url']}\n")


def print_cluster_terms(descriptions: list[list[tuple[str, float]]], report_handle) -> None:
    """Write the most informative terms for each cluster into the report."""
    for index, cluster in enumerate(descriptions):
        report_handle.write("\n" + "=" * 60 + "\n")
        report_handle.write(f"Cluster {index}\n")
        for term, weight in cluster:
            report_handle.write(f"{term:<20} {weight:.4f}\n")


def write_report(results: list[dict], best_result: dict, corpus: list[dict], descriptions: list[list[tuple[str, float]]]) -> None:
    """Write a text report with clustering metrics and cluster summaries."""
    with open(REPORTS / "z3_report.txt", "w", encoding="utf-8") as handle:
        handle.write("Z3 - Semantic clustering\n\n")

        for result in results:
            handle.write(
                f"k={result['k']:2d}"
                f"   WCSS={result['wcss']:.2f}"
                f"   silhouette={result['silhouette']:.4f}\n"
            )

        handle.write(f"\nBest k = {best_result['k']}\n")
        handle.write("\nCluster examples\n")
        print_cluster_examples(best_result, corpus, handle)
        handle.write("\nCluster terms\n")
        print_cluster_terms(descriptions, handle)


def main() -> None:
    """Run the clustering workflow end to end."""
    documents, corpus, vocab, svd = load_inputs()

    print("Running k-means clustering...")
    results = evaluate_clusters(documents)

    print("Saving plots...")
    save_plots(results)

    best_result = max(results, key=lambda item: item["silhouette"])
    descriptions = describe_clusters(best_result["centroids"], vocab, svd)

    write_report(results, best_result, corpus, descriptions)

    print(f"Report saved to: {REPORTS / 'z3_report.txt'}")
    print("Done!")


if __name__ == "__main__":
    main()
