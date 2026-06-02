#!/usr/bin/env python3
# Copyright (c) 2026 NTT, Inc.
# All rights reserved
# By Takuo Yamaguchi, 2026
"""
diagnostics.py - Save diagnostic data during algorithm1_2 execution
Outputs candidate error distributions and selection logs as CSV.
"""

from pathlib import Path
from typing import List, Dict

import pandas as pd

from utils import mean_jaccard_distance


def _labels_to_str(labels: list) -> str:
    """Convert label list to semicolon-separated string."""
    return ";".join(str(l) for l in sorted(labels))


def _candidate_key(c: Dict) -> tuple:
    """Generate a unique key for a candidate."""
    return (
        frozenset(c["labels"][0]),
        frozenset(c["labels"][1]),
        frozenset(c["labels"][2]),
    )


def save_candidates_csv(
    candidates: List[Dict],
    diversity_config: Dict,
    output_dir: Path,
) -> List[Dict]:
    """Save all algorithm1 candidates to CSV.

    Returns:
        sorted_candidates: Candidates sorted by total_error (for selection_log reference)
    """
    eutt_th = diversity_config["eutt_threshold"]
    ebs_th = diversity_config["ebs_threshold"]

    sorted_cand = sorted(candidates, key=lambda x: x["error"]["total_error"])

    rows = []
    for i, c in enumerate(sorted_cand):
        e = c["error"]
        eutt = e["eutt"]
        ebs = e["ebs"]

        passes_eutt = eutt_th is None or eutt <= eutt_th
        passes_ebs = ebs_th is None or ebs <= ebs_th

        rows.append(
            {
                "candidate_id": i,
                "eutt": eutt,
                "ebs": ebs,
                "total_error": e["total_error"],
                "train_ratio": e.get("train_ratio", ""),
                "dev_ratio": e.get("dev_ratio", ""),
                "eval_ratio": e.get("eval_ratio", ""),
                "passes_eutt_filter": passes_eutt,
                "passes_ebs_filter": passes_ebs,
                "passes_error_filter": passes_eutt and passes_ebs,
                "Ltrain": _labels_to_str(c["labels"][0]),
                "Ldev": _labels_to_str(c["labels"][1]),
                "Leval": _labels_to_str(c["labels"][2]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "all_candidates.csv", index=False)

    return sorted_cand


def save_selection_log_csv(
    selected: List[Dict],
    sorted_candidates: List[Dict],
    output_dir: Path,
) -> None:
    """Save algorithm2 selection log to CSV."""
    if not selected:
        pd.DataFrame(
            columns=[
                "step",
                "candidate_id",
                "eutt",
                "ebs",
                "total_error",
                "mean_jaccard_distance",
                "Ltrain",
                "Ldev",
                "Leval",
            ]
        ).to_csv(output_dir / "selection_log.csv", index=False)
        return

    # Build reverse lookup map for candidate IDs
    key_to_id = {}
    for i, c in enumerate(sorted_candidates):
        key_to_id[_candidate_key(c)] = i

    rows = []
    selected_labels_so_far = []
    for step, s in enumerate(selected):
        cid = key_to_id.get(_candidate_key(s), -1)
        mjd = mean_jaccard_distance(s["labels"], selected_labels_so_far)

        rows.append(
            {
                "step": step,
                "candidate_id": cid,
                "eutt": s["error"]["eutt"],
                "ebs": s["error"]["ebs"],
                "total_error": s["error"]["total_error"],
                "mean_jaccard_distance": mjd,
                "Ltrain": _labels_to_str(s["labels"][0]),
                "Ldev": _labels_to_str(s["labels"][1]),
                "Leval": _labels_to_str(s["labels"][2]),
            }
        )
        selected_labels_so_far.append(s["labels"])

    pd.DataFrame(rows).to_csv(output_dir / "selection_log.csv", index=False)


def save_diagnostics_summary(
    condition: str,
    diversity_config: Dict,
    candidates: List[Dict],
    selected: List[Dict],
    n_select: int,
    output_dir: Path,
) -> None:
    """Save a human-readable diagnostics summary."""
    eutt_th = diversity_config["eutt_threshold"]
    ebs_th = diversity_config["ebs_threshold"]
    dtype = diversity_config["type"]

    total = len(candidates)
    if total == 0:
        with open(output_dir / "diagnostics_summary.txt", "w") as f:
            f.write(f"Condition: {condition}\n")
            f.write(f"Total candidates from algorithm1: 0\n")
        return

    eutt_vals = [c["error"]["eutt"] for c in candidates]
    ebs_vals = [c["error"]["ebs"] for c in candidates]

    passes_eutt = sum(1 for v in eutt_vals if eutt_th is None or v <= eutt_th)
    passes_ebs = sum(1 for v in ebs_vals if ebs_th is None or v <= ebs_th)
    passes_both = sum(
        1
        for c in candidates
        if (eutt_th is None or c["error"]["eutt"] <= eutt_th)
        and (ebs_th is None or c["error"]["ebs"] <= ebs_th)
    )

    with open(output_dir / "diagnostics_summary.txt", "w") as f:
        f.write(f"Condition: {condition}\n")
        f.write(f"Algorithm: algorithm1_2\n")
        f.write(f"Diversity type: {dtype}\n")
        f.write(f"Thresholds: eutt<={eutt_th}, ebs<={ebs_th}\n")
        f.write(f"\n")
        f.write(f"Total candidates from algorithm1: {total}\n")
        f.write(
            f"Candidates passing eutt filter: {passes_eutt} ({passes_eutt / total * 100:.2f}%)\n"
        )
        f.write(
            f"Candidates passing ebs filter: {passes_ebs} ({passes_ebs / total * 100:.2f}%)\n"
        )
        f.write(
            f"Candidates passing both filters: {passes_both} ({passes_both / total * 100:.2f}%)\n"
        )
        f.write(f"\n")
        f.write(f"Min eutt: {min(eutt_vals):.6f}\n")
        f.write(f"Min ebs: {min(ebs_vals):.6f}\n")
        f.write(
            f"Min total_error: {min(c['error']['total_error'] for c in candidates):.6f}\n"
        )
        f.write(f"\n")
        f.write(f"Selected by algorithm2: {len(selected)} / {n_select}\n")


def save_algorithm1_2_diagnostics(
    condition: str,
    candidates: List[Dict],
    selected: List[Dict],
    n_select: int,
    diversity_config: Dict,
    output_dir: str,
) -> None:
    """Save all algorithm1_2 diagnostic data at once."""
    diag_dir = Path(output_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    sorted_cand = save_candidates_csv(candidates, diversity_config, diag_dir)
    save_selection_log_csv(selected, sorted_cand, diag_dir)
    save_diagnostics_summary(
        condition, diversity_config, candidates, selected, n_select, diag_dir
    )

    print(f"  Diagnostics saved: {diag_dir}")
