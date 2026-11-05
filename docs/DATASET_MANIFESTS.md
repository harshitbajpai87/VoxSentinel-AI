# Dataset Manifests

## Executed in this repository
`training/datasets/manifest.py` builds deterministic, leakage-checked manifests and speaker-disjoint
splits. It was run against the real local LibriSpeech `test-clean` copy:

```
python3 training/datasets/manifest.py librispeech --root backend/data/LibriSpeech
{"rows": 5323, "splits": {"train": 4247, "val": 570, "test": 506},
 "speakers": 80, "seed": "voxsentinel-2026", "speaker_disjoint": true}
```

Outputs checked in:
- `training/datasets/manifests/librispeech_test_clean.csv` —
  `path, sha256, speaker_id, label, source, license, duration_s, split`
- `training/datasets/manifests/librispeech_test_clean.summary.json`

## Guarantees enforced by the script
1. **Deterministic splits** — `sha256(seed + path)` bucketed 70/15/15, identical on any machine.
2. **Speaker-disjoint** — each speaker appears in exactly one split.
3. **Duplicate detection** — content SHA-256 compared before splitting; duplicates crossing splits abort.
4. **Leakage assertions** — the run fails loudly on any speaker or byte-identical file crossing splits.
5. **Provenance per row** — source and licence travel with every clip.

## Prepared but not produced (no data in this environment)
| Dataset | Why not produced | How to produce |
|---------|------------------|----------------|
| ASVspoof 2019 LA / 2021 DF | access by request | use official protocol split files; do not re-split |
| Common Voice 17 (hi, en-IN) | gated, not downloaded | download, split by `client_id` |
| FLEURS hi | not downloaded | split by `speaker_id` |
| Kathbath / IndicSUPERB | not downloaded | official splits |
| VoxCeleb1/2 trials | audio not redistributable | official trial lists for EER |
| Hinglish code-mixed corpus | **no public corpus exists** | collect with consent, record language mix per row |
| Internal scam transcripts | not collected | required before any supervised fraud classifier |

None of these is claimed as downloaded, trained on, or evaluated.
