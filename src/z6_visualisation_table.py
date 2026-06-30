import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.preprocessing import normalize

from z4_communities import load_graph, detect_communities
from z3_k_means import my_kmeans


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
# URL -> index
# --------------------------------------------------

def build_url_index(corpus):
    return {doc["url"]: i for i, doc in enumerate(corpus)}


# --------------------------------------------------
# TOP TERMS PER CLUSTER (simple TF)
# --------------------------------------------------

def top_terms_for_cluster(cluster_docs, top_n=5):
    counter = Counter()

    for doc in cluster_docs:
        counter.update(doc["tokens"])

    return counter.most_common(top_n)


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------

def build_summary_table(G, corpus, X, k=15):

    # -------------------------
    # CLUSTERING SEMANTICZNY
    # -------------------------
    labels, centroids, _ = my_kmeans(X, k)

    # -------------------------
    # COMMUNITY DETECTION
    # -------------------------
    communities = detect_communities(G)

    node_to_comm = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = cid

    url_to_idx = build_url_index(corpus)

    # przypisanie community do dokumentów
    doc_communities = []

    for doc in corpus:
        url = doc["url"]

        comm = node_to_comm.get(url, -1)
        doc_communities.append(comm)

    # -------------------------
    # ANALIZA KLASTRÓW
    # -------------------------

    table = []

    for cluster_id in range(k):

        # dokumenty w klastrze
        idxs = np.where(labels == cluster_id)[0]

        cluster_docs = [corpus[i] for i in idxs]
        cluster_comms = [doc_communities[i] for i in idxs]

        # dominująca community
        comm_counter = Counter(cluster_comms)
        dominant_comm, dominant_count = comm_counter.most_common(1)[0]

        # % zgodności
        total = len(cluster_comms)
        if total > 0:
            consistency = dominant_count / total
        else:
            consistency = 0.0

        # top-5 termów
        top_terms = top_terms_for_cluster(cluster_docs, top_n=5)

        # przykładowe URL-e (max 5)
        sample_urls = [doc["url"] for doc in cluster_docs[:5]]

        table.append({
            "cluster": cluster_id,
            "dominant_community": dominant_comm,
            "consistency": consistency,
            "top_terms": top_terms,
            "sample_urls": sample_urls,
            "size": len(cluster_docs)
        })

    return table


# --------------------------------------------------
# EXPORT REPORT
# --------------------------------------------------

def save_table(table, path):

    with open(path, "w", encoding="utf-8") as f:

        f.write("Z6 — TABLE: Semantic clusters vs communities\n")
        f.write("=" * 70 + "\n\n")

        for row in table:

            f.write(f"CLUSTER {row['cluster']}\n")
            f.write(f"Size: {row['size']}\n")
            f.write(f"Dominant community: {row['dominant_community']}\n")
            f.write(f"Consistency: {row['consistency']:.3f}\n\n")

            f.write("Top terms:\n")
            for term, freq in row["top_terms"]:
                f.write(f"  {term:<20} {freq}\n")

            f.write("\nSample URLs:\n")
            for url in row["sample_urls"]:
                f.write(f"  - {url}\n")

            f.write("\n" + "-" * 70 + "\n\n")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading data...")
    corpus, X = load_data()

    print("Loading graph...")
    G = load_graph()

    print("Building table...")
    table = build_summary_table(G, corpus, X, k=15)

    print("Saving report...")
    save_table(table, REPORTS / "z6_table.txt")

    print("DONE → reports/z6_table.txt")


if __name__ == "__main__":
    main()