"""Community detection on the link graph."""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

GRAPH_JSON = RAW / "graph.json"
CORPUS_JSON = PROCESSED / "corpus.json"  # do opisu tematycznego
RANDOM_STATE = 42


def load_graph() -> nx.Graph:
    with open(GRAPH_JSON, encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for page in data:
        src = page["Url"]
        for dst in page["OutLinks"]:
            G.add_edge(src, dst)

    return G.to_undirected()


def load_corpus_lookup() -> dict[str, list[str]]:
    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)
    return {doc["url"]: doc["tokens"] for doc in corpus}


def detect_communities(G: nx.Graph):
    communities = louvain_communities(G, seed=RANDOM_STATE)
    return sorted(communities, key=len, reverse=True)


def save_size_histogram(communities) -> None:
    sizes = [len(c) for c in communities]
    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=30)
    plt.xlabel("Community size")
    plt.ylabel("Count")
    plt.title("Community size distribution")
    plt.tight_layout()
    plt.savefig(REPORTS / "z4_community_sizes.png")
    plt.close()


def top_terms_for_community(urls, lookup, top_n=15):
    counter = Counter()
    for url in urls:
        counter.update(lookup.get(url, []))
    return counter.most_common(top_n)


def write_report(G, communities, lookup) -> None:
    Q = modularity(G, communities)
    sizes = [len(c) for c in communities]

    with open(REPORTS / "z4_report.txt", "w", encoding="utf-8") as f:
        f.write("Z4 - Community detection (Louvain)\n\n")
        f.write(f"Number of communities: {len(communities)}\n")
        f.write(f"Modularity Q: {Q:.4f}\n")
        f.write(f"Largest community sizes: {sizes[:10]}\n\n")

        for i, community in enumerate(communities[:3]):
            f.write("=" * 60 + "\n")
            f.write(f"COMMUNITY {i} (size={len(community)})\n")
            urls = list(community)
            for url in urls[:15]:
                f.write(f"  {url}\n")

            terms = top_terms_for_community(urls, lookup)
            f.write("\nTop terms:\n")
            for term, count in terms:
                f.write(f"  {term:<20} {count}\n")
            f.write("\n")


def main() -> None:
    print("Loading graph...")
    G = load_graph()
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("Loading corpus for topical description...")
    lookup = load_corpus_lookup()

    print("Detecting communities (Louvain)...")
    communities = detect_communities(G)

    print("Saving histogram...")
    save_size_histogram(communities)

    print("Writing report...")
    write_report(G, communities, lookup)

    print(f"Report saved to: {REPORTS / 'z4_report.txt'}")


if __name__ == "__main__":
    main()