import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from src.z3_k_means import my_kmeans
from src.z4_communities import load_graph, detect_communities

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

LSI = PROCESSED / "lsi_documents.npy"
CORPUS = PROCESSED / "corpus.json"

KS = [3, 5, 7, 10, 15]
BEST_K_FOR_HEATMAP = 7  # which k to use for the confusion-matrix / case-study sections
TOP_COMMUNITIES_FOR_HEATMAP = 20  # trim the heatmap to the N largest communities


# ---------------------------------------------------------------------------
# Loading the building blocks (reusing Z3 / Z4 directly, see imports above)
# ---------------------------------------------------------------------------

def load_semantic_inputs():
    """Load LSI embeddings and corpus metadata (same as Z3)."""
    documents = np.load(LSI)
    documents = normalize(documents, norm="l2")

    with open(CORPUS, encoding="utf-8") as handle:
        corpus = json.load(handle)

    return documents, corpus


# ---------------------------------------------------------------------------
# Aligning the two partitions on a common set of URLs
# ---------------------------------------------------------------------------

def build_alignment(corpus, cluster_labels, communities):
    """Map every URL present in BOTH the semantic clustering and the
    community detection to its (cluster_label, community_label) pair."""
    url_to_cluster = {doc["url"]: int(label) for doc, label in zip(corpus, cluster_labels)}
    url_to_community = {url: i for i, comm in enumerate(communities) for url in comm}

    common_urls = sorted(set(url_to_cluster) & set(url_to_community))
    cluster_vec = [url_to_cluster[u] for u in common_urls]
    community_vec = [url_to_community[u] for u in common_urls]

    return common_urls, cluster_vec, community_vec


def compute_nmi(u, v):
    n = len(u)

    # Prawdopodobieństwa przynależności do klastrów i communities
    counts_u = Counter(u)
    P_u = {}
    for i, c in counts_u.items():
        P_u[i] = c / n

    counts_v = Counter(v)
    P_v = {}
    for i, c in counts_v.items():
        P_v[i] = c / n

    # P(i,j) - rozkład wspólny
    # zip(labels_u, labels_v) tworzy parę etykiet dla KAŻDEGO dokumentu z osobna,
    # Counter zlicza, ile razy każda taka para się powtórzyła
    joint_counts = Counter(zip(u, v))
    P_uv = {}
    for pair, c in joint_counts.items():
        P_uv[pair] = c / n

    I = 0.0
    for (i, j), p_ij in P_uv.items():
        I += p_ij * np.log(p_ij / (P_u[i] * P_v[j]))

    # Entropie
    H_u = -sum(p * np.log(p) for p in P_u.values())
    H_v = -sum(p * np.log(p) for p in P_v.values())

    # Normalizacja do [0,1]
    return 2 * I / (H_u + H_v)

from collections import Counter
from math import comb


def compute_ari(u, v):
    n = len(u)
    assert n == len(v)

    # tablica kontyngencji: ile dokumentów mają klastery i community
    contingency = Counter(zip(u, v))

    # sumy po wierszach (rozmiary klastrów w U)
    counts_u = Counter(u)
    # sumy po kolumnach (rozmiary communities w V)
    counts_v = Counter(v)

    # liczba par dokumentów zgodnych "razem-razem" (w tej samej komórce tabeli)
    sum_comb_c = sum(comb(c, 2) for c in contingency.values())

    # liczba par dokumentów będących razem w tym samym klastrze U
    sum_comb_a = sum(comb(c, 2) for c in counts_u.values())

    # liczba par dokumentów będących razem w tej samej community V
    sum_comb_b = sum(comb(c, 2) for c in counts_v.values())

    # wszystkie możliwe pary dokumentów
    total_pairs = comb(n, 2)

    # oczekiwana liczba zgodnych par przy podziałach losowych
    expected_index = (sum_comb_a * sum_comb_b) / total_pairs

    # maksymalna możliwa liczba zgodnych par przy podziałach identyczne
    max_index = 0.5 * (sum_comb_a + sum_comb_b)

    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0  # przypadek brzegowy - oba podziały trywialne (np. 1 grupa)

    ari = (sum_comb_c - expected_index) / denominator
    return ari

def evaluate_agreement_across_k(documents, corpus, communities) -> list[dict]:
    results = []

    for k in KS:
        labels, _, _ = my_kmeans(documents, k)
        common_urls, cluster_vec, community_vec = build_alignment(corpus, labels, communities)

        nmi = compute_nmi(cluster_vec, community_vec)
        ari = compute_ari(cluster_vec, community_vec)

        results.append({
            "k": k,
            "n_common": len(common_urls),
            "nmi": nmi,
            "ari": ari,
        })

        print(f"k={k:2d}   n_common={len(common_urls):5d}   NMI={nmi:.4f}   ARI={ari:.4f}")

    return results


def save_agreement_plot(results: list[dict]) -> None:
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
    plt.savefig(REPORTS / "z5_agreement_vs_k.png")
    plt.close()


def confusion_matrix(cluster_vec, community_vec, n_clusters, n_communities):
    matrix = np.zeros((n_communities, n_clusters), dtype=int)
    for c, k in zip(community_vec, cluster_vec):
        matrix[c, k] += 1
    return matrix


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


# ---------------------------------------------------------------------------
# Case studies: agreement / disagreement examples
# ---------------------------------------------------------------------------

def find_agreement_examples(common_urls, cluster_vec, community_vec, n=3):
    """Pairs (community, cluster) that share several documents -> strong agreement."""
    grouped: dict[tuple[int, int], list[str]] = {}
    for url, c, k in zip(common_urls, community_vec, cluster_vec):
        grouped.setdefault((c, k), []).append(url)

    agreements = [(key, urls) for key, urls in grouped.items() if len(urls) >= 2]
    agreements.sort(key=lambda item: -len(item[1]))
    return agreements[:n]


def find_disagreement_examples(common_urls, cluster_vec, community_vec, n=3):
    """Semantic clusters that are scattered across many different communities."""
    by_cluster: dict[int, dict[int, list[str]]] = {}
    for url, c, k in zip(common_urls, community_vec, cluster_vec):
        by_cluster.setdefault(k, {}).setdefault(c, []).append(url)

    candidates = [
        (k, comms) for k, comms in by_cluster.items() if len(comms) > 1
    ]
    candidates.sort(key=lambda item: -len(item[1]))
    return candidates[:n]


# ---------------------------------------------------------------------------
# Navigation hubs: high-degree nodes, semantically heterogeneous
# ---------------------------------------------------------------------------

def find_navigation_hubs(G, url_to_community, url_to_cluster, top_n=15):
    degrees = dict(G.degree())

    candidates = sorted(
        ((url, deg) for url, deg in degrees.items() if url in url_to_cluster),
        key=lambda item: -item[1],
    )[:top_n]

    hubs = [
        {
            "url": url,
            "degree": deg,
            "community": url_to_community.get(url),
            "cluster": url_to_cluster.get(url),
        }
        for url, deg in candidates
    ]
    return hubs


def summarize_hub_heterogeneity(hubs: list[dict]) -> str:
    clusters_used = {h["cluster"] for h in hubs}
    communities_used = {h["community"] for h in hubs}
    return (
        f"{len(hubs)} top-degree hubs span {len(clusters_used)} distinct "
        f"semantic clusters but only {len(communities_used)} distinct communities."
    )


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    agreement_results: list[dict],
    best_k: int,
    n_communities: int,
    agreements,
    disagreements,
    hubs,
    corpus_lookup: dict[str, list[str]],
) -> None:
    with open(REPORTS / "z5_report.txt", "w", encoding="utf-8") as f:
        f.write("Z5 - Semantic vs structural comparison\n\n")

        f.write("NMI / ARI across different k (fixed community partition):\n")
        f.write("-" * 60 + "\n")
        for r in agreement_results:
            f.write(
                f"k={r['k']:2d}   n_common={r['n_common']:5d}   "
                f"NMI={r['nmi']:.4f}   ARI={r['ari']:.4f}\n"
            )

        f.write(f"\nDetailed analysis below uses k={best_k} "
                f"against {n_communities} communities.\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("AGREEMENT EXAMPLES (same cluster AND same community)\n")
        f.write("=" * 60 + "\n")
        for (community, cluster), urls in agreements:
            f.write(f"\nCommunity {community} / Cluster {cluster} "
                     f"({len(urls)} shared documents)\n")
            for url in urls[:5]:
                f.write(f"  {url}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("DISAGREEMENT EXAMPLES (same semantic cluster, scattered communities)\n")
        f.write("=" * 60 + "\n")
        for cluster, comms in disagreements:
            f.write(f"\nCluster {cluster} spans {len(comms)} different communities\n")
            for community, urls in sorted(comms.items(), key=lambda x: -len(x[1]))[:5]:
                f.write(f"  Community {community}: {len(urls)} doc(s), e.g. {urls[0]}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("NAVIGATION HUBS (high degree, possibly semantically heterogeneous)\n")
        f.write("=" * 60 + "\n")
        f.write(summarize_hub_heterogeneity(hubs) + "\n\n")
        for hub in hubs:
            terms = Counter(corpus_lookup.get(hub["url"], [])).most_common(5)
            terms_str = ", ".join(t for t, _ in terms) if terms else "(no corpus tokens)"
            f.write(
                f"  degree={hub['degree']:5d}  community={hub['community']:3}  "
                f"cluster={hub['cluster']}  url={hub['url']}\n"
                f"      top terms: {terms_str}\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading semantic clustering inputs (LSI documents, corpus)...")
    documents, corpus = load_semantic_inputs()
    corpus_lookup = {doc["url"]: doc["tokens"] for doc in corpus}

    print("Loading graph and detecting communities...")
    G = load_graph()
    communities = detect_communities(G)
    print(f"Found {len(communities)} communities.")

    print("\nEvaluating NMI / ARI across different k...")
    agreement_results = evaluate_agreement_across_k(documents, corpus, communities)
    save_agreement_plot(agreement_results)

    print(f"\nBuilding detailed comparison for k={BEST_K_FOR_HEATMAP}...")
    labels, _, _ = my_kmeans(documents, BEST_K_FOR_HEATMAP)
    common_urls, cluster_vec, community_vec = build_alignment(corpus, labels, communities)

    matrix = confusion_matrix(
        cluster_vec, community_vec, BEST_K_FOR_HEATMAP, len(communities)
    )
    save_confusion_heatmap(matrix, TOP_COMMUNITIES_FOR_HEATMAP, REPORTS / "z5_confusion.png")

    print("Finding agreement / disagreement examples...")
    agreements = find_agreement_examples(common_urls, cluster_vec, community_vec)
    disagreements = find_disagreement_examples(common_urls, cluster_vec, community_vec)

    print("Identifying navigation hubs...")
    url_to_community = dict(zip(common_urls, community_vec))
    url_to_cluster = dict(zip(common_urls, cluster_vec))
    hubs = find_navigation_hubs(G, url_to_community, url_to_cluster)

    print("Writing report...")
    write_report(
        agreement_results,
        BEST_K_FOR_HEATMAP,
        len(communities),
        agreements,
        disagreements,
        hubs,
        corpus_lookup,
    )

    print(f"Report saved to: {REPORTS / 'z5_report.txt'}")
    print(f"Plots saved to: {REPORTS / 'z5_agreement_vs_k.png'}, {REPORTS / 'z5_confusion.png'}")
    print("Done!")


if __name__ == "__main__":
    main()