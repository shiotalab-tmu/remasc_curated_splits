# ReMASC Curated Splits

## About

This repository provides metadata and programs for a redesigned evaluation framework based on the [ReMASC](https://www.isca-archive.org/interspeech_2019/gong19_interspeech.html) corpus, enabling fair comparison across recording devices and independent analysis of individual recording conditions.

## Repository Structure

```
remasc_curated_splits/
├── curated_metadata/          # Cleaned and split metadata
│   ├── README.md              #   Documentation
│   ├── fully_closed/          #   Fully-closed split
│   └── open_keys/             #   Partially-open splits
└── src/                       # Data cleaning & splitting programs
    └── README.md              #   Documentation
```

## Documentation

- [Metadata Documentation](curated_metadata/README.md) — Data cleaning methodology, CSV format, and split details
- [Program Documentation](src/README.md) — How to run data cleaning and splitting programs

## Citation

```bibtex
@inproceedings{yamaguchi2026remasc,
  author    = {Takuo Yamaguchi and Sayaka Shiota and Naohiro Tawara},
  title     = {Evaluation Framework for Multi-Channel Spoofing Detection Through Redesign of the {ReMASC} Corpus},
  booktitle = {IFIP TC11 International Conference on Information Security and Privacy (IFIP SEC 2026)},
  year      = {2026},
}
```

## References

- ReMASC corpus: [IEEE DataPort](https://ieee-dataport.org/open-access/remasc-realistic-replay-attack-corpus-voice-controlled-systems)
- Original paper: Gong *et al.*, Interspeech 2019
