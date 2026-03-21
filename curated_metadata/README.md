# Metadata Documentation

This document describes the data cleaning and splitting methodology for the redesigned ReMASC corpus and the details of the provided metadata. For full details, please refer to the [paper](../README.md#citation).

## About the ReMASC Corpus

[ReMASC](https://www.isca-archive.org/interspeech_2019/gong19_interspeech.html) (Realistic Replay Attack Microphone Array Speech Corpus) is a large-scale audio corpus for replay spoofing detection in voice-controlled devices. It includes the following recording conditions:

| Recording Condition | Description |
|:---------------------|:------------|
| Recording Environment | Env1 (outdoor), Env2 (quiet room), Env3 (room with background music), Env4 (inside a car) |
| Recording Device | D1 (2ch), D2 (4ch), D3 (6ch), D4 (7ch) |
| Playback Device | 4 types + built-in vehicular speaker (5 types total) |
| Replay Source Recorder | Tascam DR-05, iPod Touch, TTS |
| Speaker | 50 humans + TTS (51 labels total) |
| Position Information | Varies by recording environment (relative position between microphone and sound source) |

Audio data is available from [IEEE DataPort](https://ieee-dataport.org/open-access/remasc-realistic-replay-attack-corpus-voice-controlled-systems).

## Problems with Existing Subsets

The existing subsets of the ReMASC corpus have the following issues:

1. **Inconsistent composition ratios across recording devices**: The ratios of recording environments and bonafide/spoof differ across devices, making fair comparison difficult
2. **Mixed known/unknown conditions**: Conditions other than speakers (position information, recording environment, etc.) are shared across train/eval, preventing independent analysis of each condition's impact
3. **No validation data (dev) provided**: Only train and eval subsets exist, hindering overfitting suppression and parameter tuning, which also reduces experimental reproducibility

## Data Cleaning

Data cleaning is applied to the original ReMASC corpus to standardize the composition of recording conditions across recording devices. **Data cleaning is performed only once, and the resulting cleaned data is used for all subsequent data splits.**

### Excluded Data

The following data are excluded during cleaning:

- **TTS synthesized speech**: Speech from multiple synthesis engines and speaker characteristics is mixed under the same speaker label, interfering with speaker analysis
- **Recording device D1**: Missing Env2 bonafide data, making it impossible to unify conditions with other recording devices
- **Pos0 in Env4 (built-in vehicular speaker)**: A playback device unique to Env4 that contains only spoof audio

### Standardizing Data Volume

After exclusion, the data volume per condition combination is standardized across recording devices:

1. For each condition combination *c*, aggregate the number of data items *n*<sub>*c*,*d*</sub> per recording device *d*
2. Set the adopted count as *k*<sub>*c*</sub> = min(*n*<sub>*c*,*d*</sub>)
3. Exclude condition combinations where *k*<sub>*c*</sub> < *τ* (*τ* = 10) due to insufficient statistical reliability
4. Randomly extract *k*<sub>*c*</sub> items from each recording device

### Cleaning Results

| Recording Device | Original |       | After Cleaning |       |
|:------:|:--------:|:-----:|:----------:|:-----:|
|        | bonafide | spoof | bonafide   | spoof |
| D1     | 1,473    | 6,873 | 0          | 0     |
| D2     | 2,452    | 8,212 | 2,017      | 4,651 |
| D3     | 2,159    | 7,941 | 2,017      | 4,651 |
| D4     | 2,365    | 8,311 | 2,017      | 4,651 |

### Pseudo-label Assignment

Some recording conditions (playback device, replay source recorder, some position information, etc.) exist only for spoof audio and have no corresponding labels for bonafide audio. To assign bonafide audio to subsets based on conditions in partially-open splits, **pseudo-labels** are randomly assigned to bonafide audio following the condition label distribution of spoof audio.

> **Note**: Although pseudo-labels are referenced to control subset allocation for partially-open splits, the corresponding CSV columns retain their original labels.

## CSV Format

The CSV format is based on the [publicly available ReMASC corpus](https://ieee-dataport.org/open-access/remasc-realistic-replay-attack-corpus-voice-controlled-systems).

```
file_id, label, env, recorder, source_recorder, speaker, playback_device, position, duration
```

- **file_id**: File identifier
- **label**: Audio type (2: bonafide, 3: spoof)
- **env**: Recording environment
- **recorder**: Recording device
- **source_recorder**: Replay source recorder (-1 or pseudo-label for bonafide)
- **speaker**: Speaker ID
- **playback_device**: Playback device (-1 or pseudo-label for bonafide)
- **position**: Position information
- **duration**: Audio duration (seconds)

## Data Splits

This repository provides two types of data splits: **Fully-closed split** and **Partially-open split**. Each split consists of three subsets: `train`, `dev`, and `eval`. In all splits, the data volume ratio and bonafide/spoof ratio across subsets are controlled to be as uniform as possible.

### Fully-closed Split

A data split where all recording conditions are known (closed) across train/dev/eval.

- **Target ratio**: train:dev:eval = 3:1:1
- **Files**: `fully_closed/meta.{train,dev,eval}.csv`

### Partially-open Split

A set of data splits where the target recording condition is unknown (open) across subsets, while all other conditions remain known.

Three splitting algorithms are used depending on the number of condition labels:

- **Heuristic search method**: Used when the number of labels is large. Generates random label split candidates and selects based on error metrics (utterance count ratio error *e*<sub>utt</sub>, spoof ratio error *e*<sub>sratio</sub>) and diversity criteria
- **Complete enumeration combination method**: Used when the number of labels is small. Enumerates all possible label split patterns and adopts those satisfying the target ratio
- **Binary split method**: Used when only two label types exist. Assigns one label to train/dev and the other to eval

The settings and number of splits for each condition are as follows:

| Condition | \# Labels | Split Method | \# Splits |
|:----------|:------:|:------:|:------:|
| Recording environment | 4 | Complete enumeration | 12 |
| Playback device | 4 | Complete enumeration | 12 |
| Replay source recorder | 2 | Binary split | 2 |
| Speaker | 49 | Heuristic search | 15 |
| Position - Env1 | 2 | Binary split | 2 |
| Position - Env2 | 18 | Heuristic search | 15 |
| Position - Env3 | - | N/A | - |
| Position - Env4 | 6 | Heuristic search | 15 |

**Notes**:
- Playback device: The 5th type (built-in vehicular speaker) was excluded during data cleaning, so splitting is performed with 4 types
- Speaker: Spk40 was excluded due to insufficient data (|L| = 49)
- Position (Env3): Excluded from partially-open splits as position information is fixed

Files for each split are located under `open_keys/{condition_name}/`. Position splits are conducted per recording environment and located under `open_keys/position/{env}/`.
