# Data Cleaning & Splitting Programs

Programs for generating curated metadata splits from the ReMASC corpus.
All processing steps are configured via a single YAML config file.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Data Placement

Download the [ReMASC corpus](https://ieee-dataport.org/open-access/remasc-realistic-replay-attack-corpus-voice-controlled-systems) and place it under `data/` at the repository root:

```
remasc_curated_splits/
├── data/
│   └── remasc/
│       └── complete/
│           └── meta.csv   ← required
├── curated_metadata/
└── src/
```

The default config expects `meta.csv` at `../data/remasc/complete/meta.csv` (relative to `src/`).
Update `common.meta_file` in the config if your path differs.

## Setup & Run

```bash
cd src
uv sync
uv run python main.py --config configs/config.yaml
```

Output is written to `../curated_metadata/` by default.
If the directory already exists, a numbered suffix is appended (e.g., `curated_metadata_1/`).

## Processing Pipeline

The program runs the following steps sequentially:

1. **Data cleaning** (`data_cleaning.py`)
   - Exclude source recordings, TTS, specified devices, and other conditions
   - Balance data counts per condition combination (threshold `tau`)
   - Assign pseudo-labels to bonafide samples for specified conditions

2. **Fully-closed split** (`splits.py`)
   - Split cleaned data into train/dev/eval (3:1:1) per condition combination
   - All conditions are known across subsets

3. **Partially-open splits** (`splits.py`, `algorithms.py`)
   - Generate splits where a target condition is unknown across subsets
   - Three algorithms available depending on the condition:
     - **Algorithm 3** (complete enumeration): For small label sets with fixed split sizes
     - **Binary split**: For two-label conditions
     - **Algorithm 1 + 2** (heuristic search + diversity selection): For large label sets

## Config

See `configs/config.yaml` for all parameters. Key sections:

- **`data_cleaning`**: Exclusion criteria, balancing threshold (tau), and pseudo-label settings
- **`fully_closed`**: Split ratio
- **`partially_open`**: Per-condition algorithm, label sets, constraints, and selection parameters

For detailed documentation on algorithms and split settings,
see [curated_metadata/README.md](../curated_metadata/README.md)
and the associated [paper](../README.md#citation) (Yamaguchi *et al.*, IFIP SEC 2026).

## Output Structure

```
curated_metadata/
├── config.yaml          # copy of config used
├── log.txt              # execution log
├── meta_cleaned.csv     # cleaned metadata (before pseudo-labels)
├── fully_closed/
│   ├── meta.train.csv
│   ├── meta.dev.csv
│   └── meta.eval.csv
└── open_keys/
    ├── environment/     # one subdir per split pattern
    ├── playback_device/
    ├── source_recorder/
    ├── speaker/
    └── position/
        ├── env1/
        ├── env2/
        └── env4/
```

## Visualization

For conditions using Algorithm 1 + 2, diagnostic plots are saved under each condition's `diagnostics/` directory.

## Reproducibility

Data cleaning is performed once, and all subsequent splits (fully-closed and partially-open) are generated from the same cleaned data. All results are fully reproducible when all random seeds are fixed to the same value (default: 0).
