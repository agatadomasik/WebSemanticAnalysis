from src.fetch_pages import load_urls, fetch_and_save, save_index, GRAPH_JSON
from src.z1_preprocessing import build_corpus, save_corpus
import src.z1_tdm as z1_tdm
import src.z2_truncated_svd as z2_svd
import src.z3_k_means as z3_kmeans
import src.z4_communities as z4_communities
import src.z5_comparison as z5_comparison
import src.z6_visualisation_clouds as z6_clouds
import src.z6_visualisation_graph as z6_graph
import src.z6_visualisation_table as z6_table
import src.z7_gmm as z7_gmm


def step(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    step("Step 1/9 — Fetching pages")
    urls = load_urls(GRAPH_JSON)
    url_to_file = fetch_and_save(urls)
    save_index(url_to_file)

    step("Step 2/9 — Preprocessing")
    corpus = build_corpus()
    save_corpus(corpus)

    step("Step 3/9 — TF-IDF matrix")
    z1_tdm.main()

    step("Step 4/9 — LSI / Truncated SVD")
    z2_svd.main()

    step("Step 5/9 — K-means clustering")
    z3_kmeans.main()

    step("Step 6/9 — Community detection")
    z4_communities.main()

    step("Step 7/9 — Semantic vs structural comparison")
    z5_comparison.main()

    step("Step 8/9 — Visualisations")
    z6_clouds.main()
    z6_graph.main()
    z6_table.main()

    step("Step 9/9 — GMM comparison")
    z7_gmm.main()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
