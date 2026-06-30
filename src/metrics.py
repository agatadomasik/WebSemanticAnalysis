from collections import Counter
from math import comb

import numpy as np


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
