#!/usr/bin/env python3
# Copyright (c) 2026 NTT, Inc.
# All rights reserved
# By Takuo Yamaguchi, 2026
"""
utils.py - Common utility functions
"""

import random
from math import comb
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set


def calculate_total_combinations(
    n_labels: int, ltrain_min: int, ltrain_max: int, ldev_fixed: int = None
) -> int:
    """Calculate total number of combinations under constraints.

    Args:
        n_labels: Total number of labels
        ltrain_min: Minimum number of train labels
        ltrain_max: Maximum number of train labels
        ldev_fixed: Fixed number of dev labels (if applicable)
    """
    total = 0
    for n_train in range(ltrain_min, ltrain_max + 1):
        n_remaining = n_labels - n_train
        if ldev_fixed:
            dev_range = [ldev_fixed]
        else:
            dev_range = range(1, n_remaining)
        for n_dev in dev_range:
            if n_remaining - n_dev >= 1:  # at least 1 label for eval
                total += comb(n_labels, n_train) * comb(n_remaining, n_dev)
    return total


def calculate_error(
    train_data: pd.DataFrame,
    dev_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    target_split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Dict[str, float]:
    """Calculate error metrics: eutt + ebs."""
    train_n = len(train_data)
    dev_n = len(dev_data)
    eval_n = len(eval_data)
    total_n = train_n + dev_n + eval_n

    if total_n == 0 or train_n == 0 or dev_n == 0 or eval_n == 0:
        return {"eutt": float("inf"), "ebs": float("inf"), "total_error": float("inf")}

    # eutt: utterance count ratio error
    actual = (train_n / total_n, dev_n / total_n, eval_n / total_n)
    eutt = sum(abs(a - t) for a, t in zip(actual, target_split_ratio))

    # ebs: bonafide/spoof ratio error
    def bona_ratio(df):
        return len(df[df["speech type id"] == 2]) / len(df) if len(df) > 0 else 0

    all_data = pd.concat([train_data, dev_data, eval_data])
    target_bona = bona_ratio(all_data)

    ebs = (
        abs(bona_ratio(train_data) - target_bona)
        + abs(bona_ratio(dev_data) - target_bona)
        + abs(bona_ratio(eval_data) - target_bona)
    )

    return {
        "eutt": eutt,
        "ebs": ebs,
        "total_error": eutt + ebs,
        "train_ratio": actual[0],
        "dev_ratio": actual[1],
        "eval_ratio": actual[2],
    }


def jaccard_distance(set1, set2) -> float:
    """Jaccard distance: 1 - |A intersection B| / |A union B|"""
    s1, s2 = set(set1), set(set2)
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return 1 - inter / union if union > 0 else 0


def check_jaccard_threshold(
    labels: Tuple[List, List, List],
    selected: List[Tuple[List, List, List]],
    threshold: float = 0.3,
) -> bool:
    """Check that Jaccard distance >= threshold for all subsets."""
    if not selected:
        return True
    for s in selected:
        if jaccard_distance(labels[0], s[0]) < threshold:
            return False
        if jaccard_distance(labels[1], s[1]) < threshold:
            return False
        if jaccard_distance(labels[2], s[2]) < threshold:
            return False
    return True


def mean_jaccard_distance(
    labels: Tuple[List, List, List],
    selected: List[Tuple[List, List, List]],
) -> float:
    """Mean Jaccard distance to already-selected splits."""
    if not selected:
        return 0.0
    dists = []
    for s in selected:
        d = (
            jaccard_distance(labels[0], s[0])
            + jaccard_distance(labels[1], s[1])
            + jaccard_distance(labels[2], s[2])
        ) / 3
        dists.append(d)
    return np.mean(dists)


def selectCandidate(
    candidates: List[Dict],
    selected: List[Dict],
    diversity_config: Dict,
) -> Optional[Dict]:
    """selectCandidate (Algorithm 2): Select candidate based on diversity criterion M.

    Args:
        candidates: Candidate list (with error info, sorted by error)
        selected: Already-selected candidates
        diversity_config: Diversity selection settings
            - type: Selection type
                - "jaccard_threshold": Sequential selection + Jaccard threshold
                - "jaccard_threshold_random": Random selection + error threshold + Jaccard threshold
                - "diversity_score": Error threshold + max mean Jaccard distance
                - "min_error_jaccard_threshold": Min error priority + Jaccard threshold
            - jaccard_min: Jaccard distance threshold (required for jaccard-based types)
            - eutt_threshold: Utterance ratio error threshold
            - ebs_threshold: Bonafide ratio error threshold
    """
    dtype = diversity_config["type"]
    selected_labels = [c["labels"] for c in selected]

    # Error filtering
    eutt_th = diversity_config["eutt_threshold"]
    ebs_th = diversity_config["ebs_threshold"]

    def passes_error_filter(c):
        if eutt_th is None and ebs_th is None:
            return True
        if "error" not in c:
            return False
        eutt = c["error"].get("eutt", 0)
        ebs = c["error"].get("ebs", 0)
        if eutt_th is not None and eutt > eutt_th:
            return False
        if ebs_th is not None and ebs > ebs_th:
            return False
        return True

    # Exclude already-selected
    remaining = [c for c in candidates if c not in selected]

    if dtype == "jaccard_threshold":
        # Sequential selection (by error) + Jaccard threshold
        threshold = diversity_config["jaccard_min"]
        for c in remaining:
            if check_jaccard_threshold(c["labels"], selected_labels, threshold):
                return c
        return None

    elif dtype == "jaccard_threshold_random":
        # Random selection + error threshold + Jaccard threshold
        threshold = diversity_config["jaccard_min"]
        filtered = [c for c in remaining if passes_error_filter(c)]
        random.shuffle(filtered)
        for c in filtered:
            if check_jaccard_threshold(c["labels"], selected_labels, threshold):
                return c
        return None

    elif dtype == "diversity_score":
        # Error threshold + max mean Jaccard distance
        filtered = [c for c in remaining if passes_error_filter(c)]

        if not filtered:
            return None
        return max(
            filtered, key=lambda c: mean_jaccard_distance(c["labels"], selected_labels)
        )

    elif dtype == "min_error_jaccard_threshold":
        # Min error priority + Jaccard threshold
        threshold = diversity_config["jaccard_min"]
        # Candidates are sorted by error, so check Jaccard condition sequentially
        for c in remaining:
            if check_jaccard_threshold(c["labels"], selected_labels, threshold):
                return c
        return None

    return None


def convert_ratio_to_counts(constraint: Dict, n_labels: int) -> Dict:
    """Convert ratio specification (rtrain etc.) to label count specification (ltrain_min etc.)."""
    result = constraint.copy()

    # If already specified as label counts, return as-is
    if "ltrain_min" in result:
        return result

    # Ratio format: all three must be present
    rtrain = result["rtrain"]
    rdev = result["rdev"]
    reval = result["reval"]

    total_r = rtrain + rdev + reval
    ltrain = round(n_labels * rtrain / total_r)
    ldev = round(n_labels * rdev / total_r)

    result["ltrain_min"] = ltrain
    result["ltrain_max"] = ltrain
    result["ldev"] = ldev

    return result
