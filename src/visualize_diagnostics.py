#!/usr/bin/env python3
"""
visualize_diagnostics.py - Visualize algorithm1_2 diagnostic data
Auto-detects diagnostics/ directories and generates plots.

Usage:
    uv run python visualize_diagnostics.py --output-dir <output directory>
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_diagnostics_dirs(output_dir: str) -> List[Tuple[str, Path]]:
    """Recursively find diagnostics/ directories and infer condition names."""
    results = []
    base = Path(output_dir)

    for diag_dir in sorted(base.rglob("diagnostics")):
        if not diag_dir.is_dir():
            continue
        if not (diag_dir / "all_candidates.csv").exists():
            continue

        # Infer condition name from path (e.g. open_keys/position/env2/diagnostics -> position/env2)
        rel = diag_dir.relative_to(base)
        parts = list(rel.parts)
        parts = [p for p in parts if p not in ("open_keys", "diagnostics")]
        condition_name = "/".join(parts) if parts else str(rel.parent)

        results.append((condition_name, diag_dir))

    return results


def plot_condition_diagnostics(
    condition_name: str,
    diagnostics_dir: Path,
    fmt: str = "png",
) -> None:
    """Generate 2x2 diagnostic plots for one condition."""
    cand_df = pd.read_csv(diagnostics_dir / "all_candidates.csv")
    sel_df = pd.read_csv(diagnostics_dir / "selection_log.csv")

    # Read thresholds from summary
    eutt_th, ebs_th = None, None
    summary_path = diagnostics_dir / "diagnostics_summary.txt"
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            if line.startswith("Thresholds:"):
                for part in line.split(","):
                    part = part.strip()
                    if "eutt<=" in part:
                        val = part.split("eutt<=")[-1]
                        eutt_th = None if val == "None" else float(val)
                    if "ebs<=" in part:
                        val = part.split("ebs<=")[-1]
                        ebs_th = None if val == "None" else float(val)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Diagnostics: {condition_name}", fontsize=14, fontweight="bold")

    # --- (0,0) eutt vs ebs scatter plot ---
    ax = axes[0, 0]

    passed = cand_df[cand_df["passes_error_filter"] == True]
    failed = cand_df[cand_df["passes_error_filter"] == False]

    ax.scatter(
        failed["eutt"],
        failed["ebs"],
        s=1,
        alpha=0.3,
        c="gray",
        label=f"Failed ({len(failed)})",
    )
    if len(passed) > 0:
        ax.scatter(
            passed["eutt"],
            passed["ebs"],
            s=5,
            alpha=0.6,
            c="blue",
            label=f"Passed ({len(passed)})",
        )
    if len(sel_df) > 0:
        ax.scatter(
            sel_df["eutt"],
            sel_df["ebs"],
            s=15,
            c="red",
            zorder=5,
            label=f"Selected ({len(sel_df)})",
        )

    if eutt_th is not None:
        ax.axvline(
            eutt_th, color="red", linestyle="--", alpha=0.7, label=f"eutt_th={eutt_th}"
        )
    if ebs_th is not None:
        ax.axhline(
            ebs_th, color="orange", linestyle="--", alpha=0.7, label=f"ebs_th={ebs_th}"
        )

    ax.set_xlabel("eutt (split ratio error)")
    ax.set_ylabel("ebs (bona ratio error)")
    ax.set_title("Error Distribution")
    ax.legend(fontsize=8, loc="upper right")

    # --- (0,1) eutt histogram ---
    ax = axes[0, 1]
    eutt_vals = cand_df["eutt"].values
    eutt_finite = eutt_vals[np.isfinite(eutt_vals)]

    if len(eutt_finite) > 0:
        ax.hist(eutt_finite, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        if eutt_th is not None:
            ax.axvline(
                eutt_th,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"threshold={eutt_th}",
            )
            n_pass = np.sum(eutt_finite <= eutt_th)
            pct = n_pass / len(eutt_finite) * 100
            ax.text(
                0.95,
                0.95,
                f"Pass: {n_pass} ({pct:.1f}%)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            ax.legend(fontsize=8)
        ax.text(
            0.95,
            0.85,
            f"min={min(eutt_finite):.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    ax.set_xlabel("eutt")
    ax.set_ylabel("Count")
    ax.set_title("eutt Distribution")

    # --- (1,0) ebs histogram ---
    ax = axes[1, 0]
    ebs_vals = cand_df["ebs"].values
    ebs_finite = ebs_vals[np.isfinite(ebs_vals)]

    if len(ebs_finite) > 0:
        ax.hist(ebs_finite, bins=100, color="coral", alpha=0.7, edgecolor="none")
        if ebs_th is not None:
            ax.axvline(
                ebs_th,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"threshold={ebs_th}",
            )
            n_pass = np.sum(ebs_finite <= ebs_th)
            pct = n_pass / len(ebs_finite) * 100
            ax.text(
                0.95,
                0.95,
                f"Pass: {n_pass} ({pct:.1f}%)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            ax.legend(fontsize=8)
        ax.text(
            0.95,
            0.85,
            f"min={min(ebs_finite):.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    ax.set_xlabel("ebs")
    ax.set_ylabel("Count")
    ax.set_title("ebs Distribution")

    # --- (1,1) Jaccard distance ---
    ax = axes[1, 1]
    if len(sel_df) > 0:
        steps = sel_df["step"].values
        mjds = sel_df["mean_jaccard_distance"].values
        ax.bar(steps, mjds, color="teal", alpha=0.7)
        ax.set_xlabel("Selection Step")
        ax.set_ylabel("Mean Jaccard Distance")
        ax.set_title("Diversity (Mean Jaccard Distance)")
        ax.set_xticks(steps)
    else:
        ax.text(
            0.5,
            0.5,
            "No candidates selected",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_title("Diversity (Mean Jaccard Distance)")

    plt.tight_layout()
    out_path = diagnostics_dir / f"diagnostics_plots.{fmt}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize algorithm1_2 diagnostic data"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Pipeline output directory"
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="Output image format (default: png)",
    )
    args = parser.parse_args()

    diagnostics_list = find_diagnostics_dirs(args.output_dir)

    if not diagnostics_list:
        print(f"No diagnostics directories found in: {args.output_dir}")
        return

    print(f"Found {len(diagnostics_list)} diagnostics directories:")
    for name, path in diagnostics_list:
        print(f"  {name}: {path}")

    for name, diag_dir in diagnostics_list:
        print(f"\nGenerating plots for: {name}")
        plot_condition_diagnostics(name, diag_dir, args.format)

    print(f"\nDone! Generated plots for {len(diagnostics_list)} conditions.")


if __name__ == "__main__":
    main()
