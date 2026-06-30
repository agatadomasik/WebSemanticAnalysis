import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz

ROOT = Path(__file__).parent.parent
CORPUS_JSON = ROOT / "data" / "processed" / "corpus.json"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TDM_OUT = PROCESSED_DIR / "tdm.npz"
VOCAB_OUT = PROCESSED_DIR / "vocab.json"
REPORT_OUT = REPORTS_DIR / "z1_report.txt"

MIN_DF = 5       # the term must appear in at least 5 documents
MAX_DF_RATIO = 0.95  # the term cannot appear in more than 95% of documents


def build_vocabulary(corpus: list[dict]) -> dict[str, int]:
    N = len(corpus)
    df_counter = Counter()

    for doc in corpus:
        for term in set(doc["tokens"]):
            df_counter[term] += 1

    filtered = [
        term for term, df in sorted(df_counter.items())
        if MIN_DF <= df <= MAX_DF_RATIO * N
    ]
    vocab = {term: idx for idx, term in enumerate(filtered)}

    print(f"Dictionary: {len(vocab)} terms (after filtering MIN_DF={MIN_DF}, MAX_DF={MAX_DF_RATIO})")
    return vocab, df_counter


def compute_tfidf(corpus: list[dict], vocab: dict[str, int], df_counter: Counter) -> csr_matrix:
    N = len(corpus)
    t = len(vocab)
    d = N

    rows, cols, values = [], [], []

    for doc_idx, doc in enumerate(corpus):
        tf_counter = Counter(doc["tokens"])

        for term, tf in tf_counter.items():
            if term not in vocab:
                continue
            term_idx = vocab[term]

            if term_idx >= t:
                continue

            df = df_counter[term]
            tfidf = tf * np.log(N / df)

            rows.append(term_idx)
            cols.append(doc_idx)
            values.append(tfidf)

    matrix = csr_matrix((values, (rows, cols)), shape=(t, d), dtype=np.float32)
    return matrix


def generate_report(matrix: csr_matrix, vocab: dict[str, int], df_counter: Counter, corpus: list[dict]) -> str:
    N = len(corpus)
    t, d = matrix.shape
    nnz = matrix.nnz
    sparsity = 100.0 * (1 - nnz / (t * d))
    nonzero_pct = 100.0 * nnz / (t * d)

    # IDF for each term: log(N / DF(t))
    idx_to_term = {v: k for k, v in vocab.items()}
    idf_scores = {
        term: np.log(N / df_counter[term])
        for term in vocab
    }

    # Top-20 according to IDF (the highest IDF - rare, specific terms)
    #top20_idf = sorted(idf_scores.items(), key=lambda x: -x[1])[:20]
    seen_idf = set()
    top20_idf = []
    for term, score in sorted(idf_scores.items(), key=lambda x: -x[1]):
        rounded = round(score, 4)
        if rounded not in seen_idf:
            seen_idf.add(rounded)
            top20_idf.append((term, score))
        if len(top20_idf) == 20:
            break

    # global TF - TF sum over all documents for each term
    tf_global = {}
    t = matrix.shape[0]
    for term, idx in vocab.items():
        if idx >= t:
            continue
        row = matrix.getrow(idx)
        tf_global[term] = float(row.sum())

    top20_tf = sorted(tf_global.items(), key=lambda x: -x[1])[:20]

    lines = [
        "=" * 60,
        "Z1 — Preprocessing and TF-IDF matrix",
        "=" * 60,
        "",
        f"Documents number (d):        {d}",
        f"Terms number (t):            {t}",
        f"Matrix size (t×d):       {t} × {d}",
        f"Number of non-zero values:  {nnz}",
        f"% non-zero:                {nonzero_pct:.4f}%",
        f"% zero (sparsity):        {sparsity:.4f}%",
        "",
        "-" * 60,
        "TOP-20 terms according to IDF (rarest/most specific):",
        "-" * 60,
    ]
    for rank, (term, score) in enumerate(top20_idf, 1):
        lines.append(f"  {rank:2d}. {term:<25} IDF = {score:.4f}")

    lines += [
        "",
        "-" * 60,
        "TOP-20 terms by TF sum (most frequently occurring):",
        "-" * 60,
    ]
    for rank, (term, score) in enumerate(top20_tf, 1):
        lines.append(f"  {rank:2d}. {term:<25} TF_sum = {score:.1f}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


def main():
    print("Loading corpus...")
    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Documents in corpus: {len(corpus)}")

    print("\nBuilding dictionary...")
    vocab, df_counter = build_vocabulary(corpus)

    if len(vocab) < 1000:
        print(f"UWAGA: słownik ma tylko {len(vocab)} termów — poniżej wymaganego minimum 1000.")
        print("Rozważ zmniejszenie MIN_DF lub odrzucenie mniejszej liczby stop words.")

    print("\nComputing TF-IDF...")
    matrix = compute_tfidf(corpus, vocab, df_counter)
    print(f"Matrix: {matrix.shape[0]} × {matrix.shape[1]}, nnz={matrix.nnz}")

    print("\nSaving results...")
    save_npz(TDM_OUT, matrix)
    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  TDM  → {TDM_OUT}")
    print(f"  vocab → {VOCAB_OUT}")

    print("\nGenerating report...")
    report = generate_report(matrix, vocab, df_counter, corpus)
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(f"  raport: {REPORT_OUT}")
    print()
    print(report)


if __name__ == "__main__":
    main()