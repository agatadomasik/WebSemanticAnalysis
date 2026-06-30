import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.preprocessing import normalize

# jeśli masz własny kmeans
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
# CLUSTERING (ODTWORZENIE LABELS)
# --------------------------------------------------

def get_labels(X, k=15):
    labels, centroids, inertia = my_kmeans(X, k)
    return labels


# --------------------------------------------------
# TOP CLUSTERS
# --------------------------------------------------

def get_top_clusters(labels, top_n=3):
    counts = Counter(labels)
    return [c for c, _ in counts.most_common(top_n)]


# --------------------------------------------------
# BUILD TEXT PER CLUSTER
# --------------------------------------------------

def build_cluster_texts(corpus, labels):
    cluster_tokens = defaultdict(list)

    for doc, label in zip(corpus, labels):
        cluster_tokens[label].extend(doc["tokens"])

    return cluster_tokens


# --------------------------------------------------
# WORDCLOUD
# --------------------------------------------------

def create_wordcloud(tokens, cluster_id):
    text = " ".join(tokens)

    wc = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        max_words=120,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Cluster {cluster_id}")

    out_path = REPORTS / f"z6_wordcloud_cluster_{cluster_id}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("Loading data...")
    corpus, X = load_data()

    print("Clustering (k-means)...")
    labels = get_labels(X, k=15)

    print("Building clusters...")
    cluster_tokens = build_cluster_texts(corpus, labels)

    print("Selecting top clusters...")
    top_clusters = get_top_clusters(labels, 3)

    print(f"Top clusters: {top_clusters}")

    for cid in top_clusters:
        print(f"Generating wordcloud for cluster {cid}")
        create_wordcloud(cluster_tokens[cid], cid)


if __name__ == "__main__":
    main()