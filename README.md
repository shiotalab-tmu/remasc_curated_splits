# ReMASC Curated Splits

## About

このリポジトリでは，ReMASCコーパスを再設計し，収録機器間での公平な比較と各収録条件の独立した分析を可能にする評価基盤のメタデータを提供します．

### ReMASCコーパス

ReMASC (Realistic Replay Attack Microphone Array Speech Corpus) [[Gong *et al.*, 2019]](https://www.isca-archive.org/interspeech_2019/gong19_interspeech.html) は，音声操作デバイスにおける録音再生なりすまし音声検出のための大規模音声コーパスです．音声操作デバイスが運用される多様な環境を想定して，マイクロフォンアレイや収録環境など様々な収録条件が設定されています．

### 既存の問題点

ReMASCコーパスの既存サブセットは，
1. 収録機器間で収録条件の構成比率が異なる
2. 話者以外の条件の既知/未知が混在している
3. 検証用データ(dev)が存在しない

といった点で，公平な比較や詳細な要因分析がしづらいという課題があります．本リポジトリでは，これらの問題を解決する新しいデータ分割を提供します．

## Data Cleaning

オリジナルのReMASCコーパスからTTS合成音声と収録機器D1を除外し，条件組合せごとのデータ量を収録機器間で統一しました．

| 収録機器 | オリジナル |       | クリーニング後 |       |
|:------:|:--------:|:-----:|:----------:|:-----:|
|        | bonafide | spoof | bonafide   | spoof |
| D1     | 1,473    | 6,873 | 0          | 0     |
| D2     | 2,452    | 8,212 | 2,017      | 5,005 |
| D3     | 2,159    | 7,941 | 2,017      | 5,005 |
| D4     | 2,365    | 8,311 | 2,017      | 5,005 |

## Data Splits

本リポジトリでは `Fully-closed split` と `Partially-open split` の2種類の方針に基づくデータ分割を提供します．各データ分割は `train`, `dev`, `eval` の3つのサブセットから構成されます．

### Fully-closed split

`train`, `dev`, `eval` の3サブセット間ですべての収録条件が既知（closed）となるデータ分割です．

- **目標比率**: train:dev:eval = 3:1:1
- **ファイル**: `fully-closed/meta.*.csv`

メタデータのフォーマットは[公開されているReMASCコーパス](https://ieee-dataport.org/open-access/remasc-realistic-replay-attack-corpus-voice-controlled-systems)のものに基づいています．

### Partially-open split

着目する収録条件がサブセット間で未知（open）となり，それ以外の収録条件は既知となるように制御したデータ分割群です．

各条件に対する分割は以下の通りです：

| 条件 | データ分割組数 | 備考 |
|:-----|:------:|:-----|
| 収録環境（environment） | 12 | - |
| 再生機器（playback_device） | 12 | Env4を除外 |
| 不正収録機器（source_recorder） | 2 | - |
| 話者（speaker） | 10 | - |
| 位置情報（position） - Env1 | 2 | - |
| 位置情報（position） - Env2 | 10 | - |
| 位置情報（position） - Env4 | 10 | - |

- **ファイル**: `partially-open/{条件名}/meta.*.csv`
- **位置情報**: 収録環境ごとに分割を実施 `partially-open/position/{収録環境}/meta.*.csv`

ラベルが2種類のみの収録条件（発話位置(Env1), 不正収録機器）では，一方のラベルを train/dev に，他方を eval に割り当てます．これにより train と dev 間では条件が既知，eval に対しては未知となります．

## Citation

```bibtex
@inproceedings{yamaguchi2026remasc,
  author    = {山口 拓生 and 塩田 さやか and 俵 直弘},
  title     = {{ReMASC}コーパスの再設計によるマルチチャネルなりすまし音声検出の評価基盤},
  booktitle = {2026 Symposium on Cryptography and Information Security (SCIS 2026)},
  year      = {2026},
  month     = {1},
  address   = {Hakodate, Japan},
  note      = {3G1-4}
}
```
