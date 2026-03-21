#!/usr/bin/env python3
"""
main.py - Entry point for metadata split generation
"""

import argparse
import sys
import yaml
from pathlib import Path

from data_cleaning import clean_data
from splits import apply_fully_closed, apply_partially_open


class Tee:
    """Write to both stdout and a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main(config_path: str):
    """Main processing pipeline."""
    print("=" * 70)
    print("ReMASC Curated Splits")
    print("=" * 70)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_base = Path(config["common"]["output_dir"])
    seed = config["common"]["random_seed"]

    # Auto-increment output directory if it already exists
    if output_base.exists():
        i = 1
        while Path(f"{output_base}_{i}").exists():
            i += 1
        output_base = Path(f"{output_base}_{i}")
        print(f"Output directory already exists, using: {output_base}")

    # Create output directory & copy config
    output_base.mkdir(parents=True)
    import shutil

    shutil.copy(config_path, output_base / "config.yaml")

    # Tee stdout to log file
    log_file = open(output_base / "log.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    # 1. Data cleaning
    cleaned_path = output_base / "meta_cleaned.csv"
    data, original_data = clean_data(config, str(cleaned_path))

    # 2. Fully-closed split
    fc_dir = output_base / "fully_closed"
    apply_fully_closed(data, original_data, config, str(fc_dir))

    # 3. Partially-open splits
    po_config = config["partially_open"]

    for condition in ["environment", "playback_device", "source_recorder", "speaker"]:
        apply_partially_open(
            data,
            original_data,
            condition,
            po_config[condition],
            str(output_base / "open_keys" / condition),
            seed,
        )

    pos_config = po_config["position"]
    for env in ["env1", "env2", "env3", "env4"]:
        apply_partially_open(
            data,
            original_data,
            env,
            pos_config[env],
            str(output_base / "open_keys" / "position" / env),
            seed,
        )

    print("\n" + "=" * 70)
    print("Done!")
    print(f"Output: {output_base}")
    print("=" * 70)

    # Restore stdout & close log file
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReMASC split generation")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config YAML file"
    )
    args = parser.parse_args()

    main(args.config)
