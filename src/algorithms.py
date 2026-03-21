#!/usr/bin/env python3
"""
algorithms.py - Splitting algorithms
"""

import time
import random
from itertools import combinations
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from utils import calculate_error, selectCandidate


def algorithm1(
    data: pd.DataFrame,
    labels: List,
    label_col: str,
    constraint_K: Dict,
    max_iterations: int = None,
    seed: int = 42,
) -> List[Dict]:
    """Algorithm 1: Candidate generation (Heuristic search - Step 1)

    Specifies the range of train label count via ltrain_min/ltrain_max.
    For fixed label count, set ltrain_min = ltrain_max.
    Includes duplicate checking; terminates automatically when all patterns are explored.

    Args:
        max_iterations: Maximum number of unique candidates to find.
            If None, exhaustive search mode (continues until all patterns are covered).
    """
    from utils import calculate_total_combinations

    random.seed(seed)
    np.random.seed(seed)

    # Target data ratio is 3:1:1
    target_data_ratio = (0.6, 0.2, 0.2)

    # Label count range
    ltrain_min = constraint_K["ltrain_min"]
    ltrain_max = constraint_K["ltrain_max"]
    ldev_fixed = constraint_K["ldev"]  # None if freely chosen

    # Total combinations (for exhaustive search completion check)
    total_combinations = calculate_total_combinations(
        len(labels), ltrain_min, ltrain_max, ldev_fixed
    )

    Pcand = []
    seen = set()  # duplicate check
    start = time.time()
    checked = 0

    while True:
        # Termination conditions
        if max_iterations and len(Pcand) >= max_iterations:  # by unique candidate count
            break
        if len(seen) >= total_combinations:
            print(f"  All {total_combinations} patterns explored")
            break

        shuffled = random.sample(labels, len(labels))

        # Randomly choose train label count from ltrain_min to ltrain_max
        n_train_labels = random.randint(ltrain_min, ltrain_max)

        # Dev label count
        remaining = len(labels) - n_train_labels
        if ldev_fixed is not None:
            n_dev_labels = ldev_fixed
        else:
            n_dev_labels = random.randint(1, remaining - 1)

        Ltrain = shuffled[:n_train_labels]
        Ldev = shuffled[n_train_labels : n_train_labels + n_dev_labels]
        Leval = shuffled[n_train_labels + n_dev_labels :]

        if not Leval:
            continue

        # Duplicate check
        key = (frozenset(Ltrain), frozenset(Ldev), frozenset(Leval))
        if key in seen:
            checked += 1
            continue
        seen.add(key)

        Dtrain = data[data[label_col].isin(Ltrain)]
        Ddev = data[data[label_col].isin(Ldev)]
        Deval = data[data[label_col].isin(Leval)]

        error = calculate_error(Dtrain, Ddev, Deval, target_data_ratio)

        Pcand.append(
            {
                "labels": (Ltrain, Ldev, Leval),
                "error": error,
            }
        )

        checked += 1
        if checked % 10000 == 0:
            print(
                f"  Checked {checked:,}, {len(Pcand)} unique candidates, {time.time() - start:.0f}s"
            )

    return Pcand


def algorithm2(
    Pcand: List[Dict],
    n_select: int,
    diversity_config: Dict,
) -> List[Dict]:
    """Algorithm 2: Data split selection (Step 2)

    Filtering is handled by the diversity criterion M in selectCandidate.
    """
    # Sort by error (for reference in selectCandidate)
    sorted_cand = sorted(Pcand, key=lambda x: x["error"]["total_error"])

    Sfinal = []

    while len(Sfinal) < n_select:
        c = selectCandidate(sorted_cand, Sfinal, diversity_config)
        if c is None:
            break
        Sfinal.append(c)

    return Sfinal


def algorithm3(
    data: pd.DataFrame,
    labels: List,
    label_col: str,
    constraint_K: Dict,
) -> List[Dict]:
    """Algorithm 3: Complete enumeration method (strict label count specification only)

    Only usable when ltrain and ldev are fixed.
    For relaxed constraints (ltrain_min etc.), use Algorithm 1.
    """
    # Only accept strict constraints
    assert "ltrain" in constraint_K or (
        "ltrain_min" in constraint_K
        and constraint_K["ltrain_min"] == constraint_K["ltrain_max"]
    ), "Algorithm 3 requires fixed ltrain (use Algorithm 1 for relaxed constraints)"
    assert "ldev" in constraint_K, "Algorithm 3 requires fixed ldev"

    # Target data ratio is 3:1:1
    target_data_ratio = (0.6, 0.2, 0.2)

    # Label counts (fixed)
    ltrain = constraint_K["ltrain_min"]
    ldev = constraint_K["ldev"]

    Stotal = []
    checked = 0
    start = time.time()

    for Ltrain in combinations(labels, ltrain):
        remaining = [l for l in labels if l not in Ltrain]

        for Ldev in combinations(remaining, ldev):
            Leval = tuple(l for l in remaining if l not in Ldev)
            if not Leval:
                continue

            Dtrain = data[data[label_col].isin(Ltrain)]
            Ddev = data[data[label_col].isin(Ldev)]
            Deval = data[data[label_col].isin(Leval)]

            error = calculate_error(Dtrain, Ddev, Deval, target_data_ratio)

            Stotal.append(
                {
                    "labels": (list(Ltrain), list(Ldev), list(Leval)),
                    "error": error,
                }
            )

            checked += 1
            if checked % 10000 == 0:
                print(
                    f"  Checked {checked:,}, {len(Stotal)} candidates, {time.time() - start:.0f}s"
                )

    return Stotal


def binary_split(
    data: pd.DataFrame,
    labels: List,
    label_col: str,
    constraint_K: Dict,
    seed: int = 0,
) -> List[Dict]:
    """Binary split method: for |L|=2 cases."""
    assert len(labels) == 2
    ratio = constraint_K["dtrain_ddev_ratio"]

    results = []
    for eval_label in labels:
        td_label = [l for l in labels if l != eval_label][0]

        Deval = data[data[label_col] == eval_label]
        Dtd = data[data[label_col] == td_label].sample(frac=1, random_state=seed)
        n = round(len(Dtd) * ratio / (ratio + 1))
        Dtrain = Dtd.iloc[:n]
        Ddev = Dtd.iloc[n:]

        error = calculate_error(Dtrain, Ddev, Deval)

        results.append(
            {
                "labels": ([td_label], [td_label], [eval_label]),
                "data": (Dtrain, Ddev, Deval),
                "error": error,
                "name": f"{td_label}_{eval_label}",
            }
        )

    return results
