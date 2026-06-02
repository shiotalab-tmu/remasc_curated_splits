#!/usr/bin/env python3
# Copyright (c) 2026 NTT, Inc.
# All rights reserved
# By Takuo Yamaguchi, 2026
"""
splits.py - Apply splits to each condition
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

from algorithms import algorithm1, algorithm2, algorithm3, binary_split
from diagnostics import save_algorithm1_2_diagnostics
from utils import convert_ratio_to_counts


def _generate_split_dirname(result: Dict) -> str:
    """Generate a human-readable directory name from label assignment."""
    if "name" in result:
        return result["name"]
    Ltrain, Ldev, Leval = result["labels"]
    parts = [
        "".join(str(int(l)) for l in Ltrain),
        "".join(str(int(l)) for l in Ldev),
        "".join(str(int(l)) for l in Leval),
    ]
    return "_".join(parts)


def save_split(
    split_data: Dict,
    output_dir: str,
    original_data: pd.DataFrame,
    pattern_id: Optional[Union[int, str]] = None,
):
    """Save split results as CSV (reverse-lookup from original data by file id)."""
    path = Path(output_dir)
    if pattern_id is not None:
        path = path / str(pattern_id)
    path.mkdir(parents=True, exist_ok=True)

    train, dev, eval_ = split_data["data"]

    # Reverse-lookup from original data by file id
    orig_train = original_data[original_data["file id"].isin(train["file id"])]
    orig_dev = original_data[original_data["file id"].isin(dev["file id"])]
    orig_eval = original_data[original_data["file id"].isin(eval_["file id"])]

    orig_train.to_csv(path / "meta.train.csv", index=False, header=False)
    orig_dev.to_csv(path / "meta.dev.csv", index=False, header=False)
    orig_eval.to_csv(path / "meta.eval.csv", index=False, header=False)

    # memo.txt
    with open(path / "memo.txt", "w") as f:
        f.write(f"Labels: {split_data.get('labels', 'N/A')}\n")
        f.write(
            f"Train: {len(orig_train)}, Dev: {len(orig_dev)}, Eval: {len(orig_eval)}\n"
        )
        if "error" in split_data:
            e = split_data["error"]
            f.write(f"Error: eutt={e['eutt']:.4f}, ebs={e['ebs']:.4f}\n")


def apply_fully_closed(
    processed_data: pd.DataFrame,
    original_data: pd.DataFrame,
    config: Dict,
    output_dir: str,
) -> Dict:
    """Fully-closed split: 3:1:1 split per condition combination."""
    print("\n" + "=" * 60)
    print("Applying Fully-closed split")
    print("=" * 60)

    ratio = config["fully_closed"]["split_ratio"]
    seed = config["common"]["random_seed"]

    from data_cleaning import CONDITION_COLUMNS

    train_list, dev_list, eval_list = [], [], []

    for _, group in processed_data.groupby(CONDITION_COLUMNS + ["recording device id"]):
        shuffled = group.sample(frac=1, random_state=seed)
        total = len(shuffled)

        n_train = round(total * ratio[0] / sum(ratio))
        n_dev = round(total * ratio[1] / sum(ratio))

        train_list.append(shuffled.iloc[:n_train])
        dev_list.append(shuffled.iloc[n_train : n_train + n_dev])
        eval_list.append(shuffled.iloc[n_train + n_dev :])

    train = pd.concat(train_list, ignore_index=True)
    dev = pd.concat(dev_list, ignore_index=True)
    eval_ = pd.concat(eval_list, ignore_index=True)

    result = {"data": (train, dev, eval_)}
    save_split(result, output_dir, original_data)

    print(f"  Train: {len(train)}, Dev: {len(dev)}, Eval: {len(eval_)}")
    return result


def apply_partially_open(
    processed_data: pd.DataFrame,
    original_data: pd.DataFrame,
    condition: str,
    condition_config: Dict,
    output_dir: str,
    seed: int = 42,
) -> List[Dict]:
    """Partially-open split: make specified condition unknown."""
    print(f"\n{'=' * 60}")
    print(f"Applying Partially-open split: {condition}")
    print(f"{'=' * 60}")

    if not condition_config["enabled"]:
        print("  Skipped (disabled)")
        return []

    algo = condition_config["algorithm"]
    labels = condition_config["labels"]
    constraint = condition_config["constraint_K"]

    data = processed_data

    # Determine label column
    label_col_map = {
        "environment": "environment id",
        "playback_device": "playback device id",
        "source_recorder": "source recorder id",
        "speaker": "speaker id",
        "position": "position id",
    }

    # Environment-specific filtering (env1, env2, env3, env4)
    if condition.startswith("env") and condition != "environment":
        env_id = int(condition.replace("env", ""))
        data = data[data["environment id"] == env_id]
        label_col = "position id"  # envX splits by position id
    else:
        label_col = label_col_map[condition]

    # Env4: normalize position id via modulo 10 (internal processing)
    if condition == "env4":
        data = data.copy()
        data[label_col] = data[label_col] % 10

    # Auto-detect labels from data
    if labels is None:
        labels = sorted(data[label_col].unique())
        labels = [l for l in labels if l != -1]

    # Convert to int for logging (also fine for processing)
    labels = [int(l) for l in labels]

    print(f"  Algorithm: {algo}")
    print(f"  Labels: {labels}")


    # Execute algorithm
    if algo == "algorithm3":
        # Convert ratio to label counts
        constraint = convert_ratio_to_counts(constraint, len(labels))
        results = algorithm3(data, labels, label_col, constraint)

    elif algo == "binary_split":
        results = binary_split(data, labels, label_col, constraint, seed)

    elif algo == "algorithm1_2":
        selection = condition_config["selection"]
        diversity = condition_config["diversity_M"]

        max_iterations = selection["max_iterations"]

        # Convert ratio to label counts
        constraint = convert_ratio_to_counts(constraint, len(labels))

        # Algorithm 1 with duplicate checking. None means exhaustive search mode
        candidates = algorithm1(
            data, labels, label_col, constraint, max_iterations, seed
        )

        n_select = selection["n_select"]
        results = algorithm2(candidates, n_select, diversity)

        # Save diagnostic data (candidate distributions, selection log)
        save_algorithm1_2_diagnostics(
            condition=condition,
            candidates=candidates,
            selected=results,
            n_select=n_select,
            diversity_config=diversity,
            output_dir=output_dir,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Regenerate data from labels (all algorithms, for memory efficiency)
    for r in results:
        if "data" not in r:
            Ltrain, Ldev, Leval = r["labels"]
            Dtrain = data[data[label_col].isin(Ltrain)]
            Ddev = data[data[label_col].isin(Ldev)]
            Deval = data[data[label_col].isin(Leval)]
            r["data"] = (Dtrain, Ddev, Deval)

    # Determine output data source
    # output_normalized: true = output with normalized values
    # output_normalized: false/unset = reverse-lookup from original data (default)
    if condition_config.get("output_normalized", False):
        output_data = data
    else:
        output_data = original_data

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        if algo in ("algorithm3", "binary_split"):
            dirname = _generate_split_dirname(r)
        else:
            dirname = i
        save_split(r, output_dir, output_data, dirname)

    print(f"  Created {len(results)} splits")
    return results
