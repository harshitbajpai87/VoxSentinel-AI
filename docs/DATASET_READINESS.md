# Dataset readiness (read-only)

`backend/scripts/check_dataset_readiness.py` audits the actual files referenced by a manifest before any training or evaluation. It uses only the Python standard library. It never downloads, decodes, generates, rewrites, deletes or copies audio, loads a model/pickle/joblib, opens the application database, or trains a speech model. Its JSON report goes to stdout; redirect it to a **new** report location if persistence is desired. Do not redirect over a manifest or another supplied artifact.

From the repository root:

```powershell
python backend/scripts/check_dataset_readiness.py --manifest evaluation/manifests/training_manifest.jsonl --dataset-dir C:/datasets/multidataset
python backend/scripts/check_dataset_readiness.py --manifest backend/data/cross_generator_v4/manifests/manifest.jsonl --dataset-dir C:/datasets/generator_v4
python backend/scripts/check_dataset_readiness.py --manifest training/datasets/manifests/librispeech_test_clean.csv --dataset-dir .
```

The dataset root is explicit, not guessed from the manifest directory. The last CSV records project-relative paths beginning `backend/data/LibriSpeech/...`, so its root is the project directory. The cross-generator builder used `/tmp/av/generator_v4` in its documented original invocation; its manifest is stored separately from the audio. Multisource scripts document an external `../data` root, and the shipped unified summary records `../../data`. Historical dataset registry `downloaded` statuses describe the originating environment, not a fresh measurement of this checkout.

## Checks and exit status

* Exit 0: the supplied manifest set passes this audit. Exit 1: JSON report contains readiness defects. Exit 2: command-line usage error.
* JSONL accepts the existing `path/partition` schema and unified `file_path/split` schema. CSV is also supported. `val/dev` map to `validation`, `test` to `eval`; real/bonafide/genuine/natural and synthetic/spoof/fake labels normalize to the two detector classes. Unknown labels fail rather than being guessed.
* All three of train, validation and eval must contain both real and synthetic examples. Counts include manifest rows even when audio is missing. Language totals and per-class language totals are reported per partition; absent language is explicitly `unknown`, not inferred.
* An explicit `holdout` value remains a fourth, separate partition, and is checked against every other partition. A holdout-only manifest necessarily fails the three-partition class requirement. **The shipped `holdout_manifest.jsonl` instead labels its rows `partition: eval`; the checker honors that metadata and does not infer a new partition from a filename.** Use the unified manifest to retain its explicit holdout distinction when auditing the complete corpus.
* Existing regular files are SHA-256 hashed in 1 MiB chunks, once per resolved path. Manifest hash fields are not trusted. Equal actual content across any two partitions fails, even with different names. Missing files are reported and never opened for hashing; `bytes_hashed` and `unique_files_hashed` disclose actual work.
* Speaker IDs must be disjoint across every partition pair, including eval. IDs are compared globally, with no implicit source exemptions. Missing/unknown speaker IDs and the shipped `source:rec-...` recording proxies fail closed: a recording identifier does not establish human-speaker separation.
* Synthetic rows need known generator IDs. Reuse across any partition pair fails the strict generator-disjoint check. A known generator on real rows is not used for this synthetic-generator check. Existing experiments may deliberately share generator families between train and validation; that can be valid for other evaluation goals, but does **not** pass this stricter readiness audit.
* Absolute/drive-relative paths, `..` components (including Windows backslashes), NUL paths and resolved symlink escapes are rejected before hashing. Paths must be contained within the explicit dataset root. Duplicate manifest entries within a partition are errors as well.

Repeat `--manifest` for non-overlapping pieces of the **same corpus/root** whose declared partition metadata preserves the intended split boundaries. Do not combine `unified_manifest.jsonl` with its derived training/holdout manifests: they describe overlapping rows, not additional audio. Audit unrelated corpora separately with their own roots.

Reports contain complete error counts and at most 20 examples per error type by default (`--max-examples N` changes the bound). Byte hashing is streamed and manifest rows are streamed, but speaker, generator, path and hash indexes remain in memory and grow with corpus size. The checker does not recursively search for alternate audio locations.

## Supplied checkout: what is actually present

Inspection on 2026-11-07 found no audio extensions under `backend/data`; that directory contains JSONL, CSV and JSON data/metadata only. `backend/data/corpus` contains the text `scam_corpus.jsonl`, not speech audio. `backend/data/LibriSpeech` is absent. The `real_clean`, `real_8k`, `tts_*`, `synthetic_commands`, `audio_deepfake`, `fake_or_real` and other audio paths referenced by the original manifests are not shipped in their expected roots here. The original manifest files therefore refer to absent external audio, not to a runnable local speech-training corpus.

Measured with the new checker against the supplied local candidate roots (`backend/data/corpus` for multisource manifests; `backend/data/cross_generator_v4` for v4):

| Manifest | Rows / valid rows | Missing referenced audio | Additional structural findings |
| --- | ---: | ---: | --- |
| training_manifest.jsonl | 35,631 / 35,631 | 35,631 | 4 overlapping generator IDs; 16,019 synthetic rows without generator IDs; 9,361 recording-proxy speaker IDs |
| holdout_manifest.jsonl | 854 / 854 | 854 | All rows declare `eval` and synthetic; 854 recording-proxy speakers; other required partition/classes absent |
| unified_manifest.jsonl | 36,489 / 36,485 | 36,485 valid rows | 4 invalid/unsupported rows; 4 overlapping generators; 16,019 synthetic rows without generator IDs; 26,211 rows without speaker IDs |
| cross_generator_v4/manifests/manifest.jsonl | 1,344 / 1,344 | 1,344 | No additional declared-identity overlap detected; audio content unavailable |

The LibriSpeech CSV was also audited with the project directory as root: 5,323 valid rows, all 5,323 audio files missing; train/validation/eval have 4,247/570/506 real rows and no synthetic examples. All languages are unspecified. No declared speaker overlap was found, but audio content remains unverified.

All five audits hashed **zero files / zero bytes**. The training manifest's actual 35,631 rows differ from the historical unified summary's reported conversion count of 35,635; the audit reports current bytes rather than trusting that summary. Counts are not additive across these overlapping derived manifests. The cross-generator split has train 840 real/120 synthetic, validation 150/24, eval 150/60; eval languages are 186 English and 24 Hindi rows.

This observation does not invalidate historical results, prove the originating data never existed, or establish any model accuracy. Do not manufacture replacement waveforms, mark text fixtures as real speech, or alter the original manifests to make readiness appear to pass. Restore the legitimately obtained external datasets to a chosen root, preserve licensing/provenance, and re-run the audit before deciding whether a split is suitable.

## Tests and limitations

```powershell
python -m pytest backend/tests/test_dataset_readiness.py -q
```

Executed using the parent `test-env/Scripts/python.exe` on 2026-11-07: **18 passed, 1 skipped in 1.54 seconds**. The skipped test was Windows symlink creation; a pre-existing FastAPI/Starlette `httpx` deprecation warning was emitted during test setup.

Tests use tiny temporary arbitrary byte files, not speech. They cover successful counts and read-only behavior, train/validation/eval leakage, held-out generator leakage, actual-content duplicate detection, missing-audio no-hash behavior, unsafe paths, absent/proxy metadata, CSV/unified aliases, invalid manifests, streamed hashing and CLI JSON/exit codes. The symlink test is skipped where Windows does not permit symlink creation.

A pass is a structural readiness check, **not** proof of representative speech, correct labels, intelligibility, sample rate/codec quality, legal permission, adequate data volume, or accuracy/generalization. Re-encoded duplicates and wrong identities require stronger provenance/audio review. Input files should not be modified concurrently: this is a snapshot audit, not a secure sandbox for hostile processes racing symlinks.
