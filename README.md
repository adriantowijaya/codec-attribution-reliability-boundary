# Configuration-Conditioned Reliability Boundaries in Blind Audio Codec-Family Attribution

THIS REPOSITORY IS A REPRODUCIBILITY PACKAGE, NOT A NEW MODEL OR BENCHMARK.

This local release candidate supports reviewer-side verification for decoded-waveform AAC/MP3/Opus codec-family attribution under the frozen operating-rate design. It contains curated source code copies, frozen configurations, manifests, result summaries, prediction files, publication source data, and a standard-library verification harness.

## Included

- Frozen scientific configurations and expected-result contract.
- Parent, split, PCM, derivative, and speech provenance manifests.
- Method artifacts for STFT log-power, MFCC, permutation, bootstrap, geometry, and frozen model identity.
- Lightweight frozen predictions, summaries, bootstrap/geometry evidence, and publication CSV source data.
- Local verification scripts and unittest gates.

## Not Included

- Third-party raw audio from Orchset, IRMAS, FSDnoisy18k, or LibriSpeech.
- Codec derivatives and large feature caches that are regeneration outputs.
- FFmpeg binaries, installers, private logs, credentials, or machine-specific paths.

## Quick Verification

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

Expected outcomes include `48 kbps full event = NO` as an expected PASS condition. The 64 kbps discovery-rate replication, 80 kbps full event, and 96 kbps full event are expected to PASS as positive events.

## Full Reproduction

Full reproduction requires lawful local access to the excluded third-party corpora and an explicit data/output root:

```bash
python scripts/reproduce_full.py --data-root /path/to/lawful/corpora --output-root /path/to/work --dry-run
```

The full orchestrator is deliberately dry-run first and does not ship raw data.

## Environment

The frozen environment anchors are documented in `environment/`, including Python 3.12.3, NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.2, scikit-learn 1.5.1, deterministic thread settings, and the frozen codec-toolchain identity record where locally available.

## Provenance, Checksums, Citation, License

Checksums are under `manifests/checksums.sha256`. Author, DOI, final repository URL, ORCID, affiliation, and license fields are pending human confirmation. No placeholder DOI is asserted here.


## Public Reproducibility Release

Status: PUBLIC REPRODUCIBILITY RELEASE.

Version: 1.0.0.

Zenodo archival publication: PENDING.

Repository URL: https://github.com/adriantowijaya/codec-attribution-reliability-boundary

Source code is licensed under BSD-3-Clause. Project-generated numerical evidence and documentation are licensed under CC-BY-4.0. Third-party materials are not redistributed and are not re-licensed.

Production DOI: [PRODUCTION DOI TO BE INSERTED AFTER PUBLICATION]
