import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize

from z4_communities import load_graph, detect_communities  # jeśli masz w module
from z3_k_means import my_kmeans   # lub sklearn fallback
# albo wklej funkcje bezpośrednio

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
PROCESSED = ROOT / "data" / "processed"
LSI_PATH = PROCESSED / "lsi_documents.npy"
CORPUS_PATH = PROCESSED / "corpus.json"

def build_node_colors_communities(G, communities):
    """
    Zwraca listę kolorów (int) dla węzłów wg Louvain communities.
    """

    node_to_comm = {}

    for comm_id, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = comm_id

    colors = [
        node_to_comm.get(node, -1)
        for node in G.nodes()
    ]

    return colors

def build_node_colors_semantic(G, corpus, labels):
    """
    Zwraca listę kolorów (int) dla węzłów wg k-means (LSI).
    """

    url_to_idx = {
        doc["url"]: i
        for i, doc in enumerate(corpus)
    }

    node_to_cluster = {}

    for node in G.nodes():
        idx = url_to_idx.get(node)

        if idx is None:
            continue

        if idx >= len(labels):
            continue

        node_to_cluster[node] = int(labels[idx])

    colors = [
        node_to_cluster.get(node, -1)
        for node in G.nodes()
    ]

    return colors

def draw_graph(G, pos, node_colors, title, path):
    """
    Rysuje graf przy użyciu wcześniej policzonego layoutu.
    """

    plt.figure(figsize=(12, 12))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap="tab20",
        node_size=6,
        alpha=0.9
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.08,
        width=0.3
    )

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(path, dpi=300)
    plt.close()

def main():
    print("Loading corpus...")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
        
    print("Loading graph...")
    valid_urls = set(doc["url"] for doc in corpus)
    G = load_graph().subgraph(valid_urls)
    
    print("Loading LSI...")
    X = np.load(LSI_PATH)
    X = normalize(X)
    
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("Detecting communities...")
    communities = detect_communities(G)
    labels, _, _ = my_kmeans(X, k=15)

    print(f"Found {len(communities)} communities")

    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    print("Drawing graph (communities)...")
    node_colors_communities = build_node_colors_communities(G, communities)
    draw_graph(G, pos, node_colors_communities, "Graph — Louvain communities", REPORTS / "z6_graph_communities.png")
    
    print("Drawing graph (semantic clusters)...")
    node_colors_semantic = build_node_colors_semantic(G, corpus, labels)
    draw_graph(G, pos, node_colors_semantic, "Graph — Semantic clusters (LSI + k-means)", REPORTS / "z6_graph_semantic.png")
    
    print("Saved: z6_graph_communities.png")


if __name__ == "__main__":
    main()