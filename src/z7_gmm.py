import json
from pathlib import Path
from collections import Counter

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture

from z4_communities import load_graph, detect_communities
from z3_k_means import my_kmeans


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

CORPUS_PATH = PROCESSED / "corpus.json"
LSI_PATH = PROCESSED / "lsi_documents.npy"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    X = np.load(LSI_PATH)
    X = normalize(X)

    return corpus, X


# --------------------------------------------------
# COMMUNITIES (ground truth structure)
# --------------------------------------------------

def get_community_labels(G, corpus):
    communities = detect_communities(G)

    node_to_comm = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = cid

    labels = []
    for doc in corpus:
        labels.append(node_to_comm.get(doc["url"], -1))

    return np.array(labels)


# --------------------------------------------------
# BASELINE: K-MEANS
# --------------------------------------------------

def run_kmeans(X, k=15):
    labels, _, inertia = my_kmeans(X, k)
    return labels


# --------------------------------------------------
# METHOD: GMM
# --------------------------------------------------

def run_gmm(X, k=15):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=42
    )
    labels = gmm.fit_predict(X)
    return labels


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(X, labels, community_labels):

    # filtruj brakujące (-1)
    mask = community_labels != -1

    X_f = X[mask]
    labels_f = labels[mask]
    comm_f = community_labels[mask]

    silhouette = silhouette_score(X_f, labels_f)

    nmi = normalized_mutual_info_score(labels_f, comm_f)
    ari = adjusted_rand_score(labels_f, comm_f)

    return silhouette, nmi, ari


# --------------------------------------------------
# TABLE EXPORT
# --------------------------------------------------

def save_report(results):
    path = REPORTS / "z7_report.txt"

    with open(path, "w", encoding="utf-8") as f:

        f.write("Z7 — COMPARISON: k-means vs GMM\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"{'Model':<15}{'Silhouette':<15}{'NMI':<15}{'ARI':<15}\n")
        f.write("-" * 60 + "\n")

        for r in results:
            f.write(
                f"{r['model']:<15}"
                f"{r['silhouette']:<15.4f}"
                f"{r['nmi']:<15.4f}"
                f"{r['ari']:<15.4f}\n"
            )

        f.write("\n\nInterpretation:\n")
        f.write(
            "GMM allows soft clustering approximation, "
            "which can better capture overlapping semantic structures "
            "compared to hard k-means assignments.\n"
        )

    print(f"Saved: {path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading data...")
    corpus, X = load_data()

    print("Loading graph / communities...")
    G = load_graph()
    community_labels = get_community_labels(G, corpus)

    print("Running k-means...")
    km_labels = run_kmeans(X, k=15)

    print("Running GMM...")
    gmm_labels = run_gmm(X, k=15)

    print("Evaluating...")

    km_sil, km_nmi, km_ari = evaluate(X, km_labels, community_labels)
    gmm_sil, gmm_nmi, gmm_ari = evaluate(X, gmm_labels, community_labels)

    results = [
        {
            "model": "k-means",
            "silhouette": km_sil,
            "nmi": km_nmi,
            "ari": km_ari
        },
        {
            "model": "GMM",
            "silhouette": gmm_sil,
            "nmi": gmm_nmi,
            "ari": gmm_ari
        }
    ]

    print("\nRESULTS:")
    for r in results:
        print(r)

    print("\nSaving report...")
    save_report(results)


if __name__ == "__main__":
    main()