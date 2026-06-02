#!/usr/bin/env python3
# Copyright (c) 2026 NTT, Inc.
# All rights reserved
# By Takuo Yamaguchi, 2026
"""
data_cleaning.py - Data cleaning pipeline
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import yaml
from pathlib import Path


META_COLUMNS = [
    "file id",
    "speech type id",
    "speaker id",
    "environment id",
    "position id",
    "source recorder id",
    "playback device id",
    "recording device id",
    "recording length",
]

CONDITION_COLUMNS = [
    "speech type id",
    "speaker id",
    "environment id",
    "position id",
    "source recorder id",
    "playback device id",
]


def load_meta(meta_path: str) -> pd.DataFrame:
    """Load metadata CSV file."""
    meta = pd.read_csv(meta_path, header=None)
    meta.columns = META_COLUMNS
    return meta


def apply_exclusion_criteria(
    meta: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """Apply exclusion criteria X.

    ReMASC dataset structure:
    - source_recordings: recording device id = -1 (original bonafide speech)
    - genuine_recordings: recording device id = 2,3,4, speech type id = 2
    - spoofed_recordings: recording device id = 2,3,4, speech type id = 3

    Only genuine_recordings and spoofed_recordings are used.
    """
    print(f"Original data count: {len(meta)}")

    dc_config = config["data_cleaning"]

    # Exclude source_recordings (recording device id == -1)
    # Only genuine_recordings/spoofed_recordings are used
    before = len(meta)
    meta = meta[meta["recording device id"] != -1]
    print(f"After excluding source_recordings: {len(meta)} (-{before - len(meta)})")

    # Exclude TTS speech (source_recorder_id == 3)
    if dc_config["exclude_tts"]:
        before = len(meta)
        meta = meta[meta["source recorder id"] != 3]
        print(f"After excluding TTS: {len(meta)} (-{before - len(meta)})")

    # Exclude recording devices (e.g. D1)
    for device_id in dc_config["exclude_devices"]:
        before = len(meta)
        meta = meta[meta["recording device id"] != device_id]
        print(f"After excluding D{device_id}: {len(meta)} (-{before - len(meta)})")

    # Additional exclusion conditions
    exclude_cond = dc_config["exclude_conditions"]
    if exclude_cond["env4_position_0"]:
        before = len(meta)
        mask = ~((meta["environment id"] == 4) & (meta["position id"] == 0))
        meta = meta[mask]
        print(f"After excluding Env4 position 0: {len(meta)} (-{before - len(meta)})")

    return meta.reset_index(drop=True)


def balance_condition_combinations(
    meta: pd.DataFrame,
    min_samples_threshold: int,
    seed: int,
) -> pd.DataFrame:
    """Balance data counts across condition combinations.

    Since source_recordings are already excluded, all data has device 2,3,4.
    For each condition combination, extract the common sample count k_c
    across all devices.
    """
    np.random.seed(seed)

    devices = sorted(meta["recording device id"].unique())
    print(f"Target devices: {devices}")

    if len(devices) == 0:
        raise ValueError("No devices found!")

    # Group by condition combinations
    grouped = meta.groupby(CONDITION_COLUMNS + ["recording device id"])
    condition_counts = grouped.size().unstack(fill_value=0)

    # Minimum data count k_c per condition combination (min across all devices)
    k_c = condition_counts.min(axis=1)

    # Exclude combinations below threshold tau
    valid_conditions = k_c[k_c >= min_samples_threshold].index
    print(
        f"Valid combinations (k_c >= {min_samples_threshold}): {len(valid_conditions)}"
    )

    if len(valid_conditions) == 0:
        print("DEBUG: k_c distribution:")
        print(k_c.value_counts().sort_index())
        raise ValueError(f"No conditions with k_c >= {min_samples_threshold}")

    # Sample k_c items from each condition
    balanced_data = []
    for condition in valid_conditions:
        k = k_c[condition]
        if k == 0:
            continue
        for device in devices:
            mask = True
            for col, val in zip(CONDITION_COLUMNS, condition):
                mask = mask & (meta[col] == val)
            mask = mask & (meta["recording device id"] == device)
            subset = meta[mask]
            if len(subset) >= k:
                sampled = subset.sample(n=k, random_state=seed)
                balanced_data.append(sampled)

    if not balanced_data:
        raise ValueError("No data remaining after balancing!")

    result = pd.concat(balanced_data, ignore_index=True)
    print(f"Balanced data count: {len(result)}")

    # Statistics
    for device in devices:
        d = result[result["recording device id"] == device]
        bona = len(d[d["speech type id"] == 2])
        spoof = len(d[d["speech type id"] == 3])
        print(f"  D{device}: Genuine={bona}, Spoof={spoof}")

    return result


def assign_virtual_labels(
    meta: pd.DataFrame,
    label_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign pseudo-labels to bonafide samples.

    References the spoof label distribution and assigns labels to bonafide
    samples in the same proportions deterministically.
    The input meta should be pre-filtered to the target scope (e.g. env1 only).
    """
    meta = meta.copy()
    np.random.seed(seed)

    print(f"\n--- assign_virtual_labels: '{label_col}' ---")

    # Original spoof label distribution
    spoof_mask = meta["speech type id"] == 3
    spoof_counts = meta.loc[spoof_mask, label_col].value_counts().sort_index()
    spoof_total = spoof_mask.sum()
    print(f"  [Spoof] Original label distribution (n={spoof_total}):")
    for label, count in spoof_counts.items():
        print(f"    {label}: {count} ({count / spoof_total:.1%})")

    # Original bonafide label distribution
    bona_mask = meta["speech type id"] == 2
    bona_counts = meta.loc[bona_mask, label_col].value_counts().sort_index()
    bona_total = bona_mask.sum()
    print(f"  [Bona] Original label distribution (n={bona_total}):")
    for label, count in bona_counts.items():
        print(f"    {label}: {count} ({count / bona_total:.1%})")

    if len(spoof_counts) == 0:
        print(f"  No valid spoof labels found for '{label_col}', skipping.")
        return meta

    n = bona_mask.sum()

    if n > 0:
        # Determine count per label (remainder assigned to the last label)
        available_labels = spoof_counts.index.tolist()
        spoof_probs = spoof_counts.values.astype(float) / spoof_counts.values.sum()
        bona_vlabel_assign_counts = np.round(spoof_probs * n).astype(int)
        bona_vlabel_assign_counts[-1] = (
            n - bona_vlabel_assign_counts[:-1].sum()
        )  # round remainder to last label

        # Shuffle target indices and assign labels sequentially
        target_idx = meta.index[bona_mask].tolist()
        np.random.shuffle(target_idx)
        pos = 0
        for label, count in zip(available_labels, bona_vlabel_assign_counts):
            meta.loc[target_idx[pos : pos + count], label_col] = label
            pos += count

        assigned_counts = meta.loc[bona_mask, label_col].value_counts().sort_index()
        print(f"  [Virtual] Assigned to {n} bonafide:")
        for label, count in assigned_counts.items():
            print(f"    {label}: {count} ({count / n:.1%})")

    return meta


def clean_data(
    config: Dict,
    output_path: Optional[str] = None,
) -> tuple:
    """Main data cleaning pipeline.

    Returns:
        (processed, original): processed DataFrame (with pseudo-labels) and original data
    """
    print("=" * 60)
    print("Data Cleaning")
    print("=" * 60)

    meta_path = config["common"]["meta_file"]
    seed = config["common"]["random_seed"]
    tau = config["data_cleaning"]["min_samples_threshold"]

    # Load metadata
    meta = load_meta(meta_path)

    # Apply exclusion criteria
    meta = apply_exclusion_criteria(meta, config)

    # Balance data counts across devices
    meta = balance_condition_combinations(meta, tau, seed)

    # Preserve state before pseudo-label assignment
    original = meta.copy()

    # Assign pseudo-labels to bonafide samples based on config.
    # For conditions where only spoof has labels (bonafide = -1),
    # assign labels proportionally so that partially-open splits
    # can maintain consistent train/dev/eval assignment across speech types.
    vl_configs = config["data_cleaning"]["virtual_labels"]
    for vl in vl_configs:
        col = vl["column"]
        filt = vl.get("filter", {})
        if filt:
            mask = pd.Series(True, index=meta.index)
            for fcol, fval in filt.items():
                mask &= meta[fcol] == fval
            if mask.sum() > 0:
                updated = assign_virtual_labels(meta[mask].copy(), col, seed)
                meta.loc[mask, col] = updated[col].values
        else:
            meta = assign_virtual_labels(meta, col, seed)

    # Save original data (before pseudo-labels)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        original.to_csv(output_path, index=False, header=False)
        print(f"Saved: {output_path}")

    print("=" * 60)
    return meta, original


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    clean_data(config, args.output)
