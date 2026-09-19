# Face De-duplication Validation

This folder contains the reproducible evaluation code and compact result artefacts used to validate the face de-duplication component of the PhD prototype on AI- and blockchain-enabled internet voting.

The experiments address three questions:

1. How should the face-match threshold be calibrated for the current InsightFace implementation?
2. How does exact 1:N FAISS identification behave as the gallery grows from roughly 1,000 to 9,000 identities?
3. Does ordinary beard/stubble appearance change materially reduce the probability of finding the same person in the election-day gallery?

The code in this folder is evaluation-oriented. It is separate from the operational Triton/FAISS services elsewhere in the repository.

## Experimental stack

The additional validation experiments use:

- Python 3.14
- InsightFace Python Library 2.0
- `raccoon_l` model package
- 512-dimensional L2-normalised face embeddings
- ONNX Runtime GPU for face inference
- FAISS `IndexFlatL2` for exact nearest-neighbour search

FAISS `IndexFlatL2` returns squared Euclidean distance. Where an experiment reports a true L2 threshold `tau_l2`, the corresponding FAISS threshold is `tau_l2 ** 2`.

The earlier deployment and throughput experiments in the wider repository use the previously established `buffalo_l`/Triton/TensorRT pipeline. The experiments here are complementary validation runs and do not replace those earlier benchmarks.

## Repository layout

```text
face_dedup_validation/
├── README.md
├── .gitignore
├── requirements.txt
├── setup_windows_py314.ps1
├── check_environment.py
├── download_celeba.py
├── download_raccoon_l.py
├── experiment_common.py
├── face_embedding_backend.py
├── calibrate_threshold_raccoon.py
├── celeba_1n_eval.py
├── celeba_1n_fpir_recalibration.py
├── celeba_rethreshold_existing_results.py
├── pgu_beard_raccoon.py
├── celeba_identity_split.json
└── results/
    ├── calibration_raccoon_l/
    ├── celeba_1n_raccoon_l/
    ├── calibration_1n_fpir_raccoon_l/
    └── pgu_beard_raccoon_l_fixed/
```

The datasets, model cache, Python virtual environment, embedding caches and large per-query result files are intentionally excluded from Git.

## Datasets

### CelebA

CelebA is used as an unconstrained large-gallery stress test. It is not treated as a webcam or Indian-electorate proxy.

The experiment uses an identity-disjoint split recorded in `celeba_identity_split.json`. Calibration identities are kept separate from the identities used for the main 1:N evaluation.

The dataset itself is not redistributed in this repository. Use `download_celeba.py` or obtain CelebA from its official distribution subject to its licence and terms.

### PGU-Face

PGU-Face is used to test robustness to ordinary beard/stubble appearance change.

The primary comparison uses:

- condition 1: clean or nearly clean-shaven;
- condition 3: beard/stubble.

Both clean-to-beard and beard-to-clean directions are evaluated. The corrected parser accepts filename variations present in the distributed dataset, including forms such as `img01.jpg`, `im01.jpg`, `img 01.jpg` and `image 01.jpg`.

The dataset itself is not redistributed in this repository.

## Main scripts

### `calibrate_threshold_raccoon.py`

Performs threshold calibration for the current `raccoon_l` embedding pipeline. It produces calibration summaries, threshold sweeps and embedding-failure records.

### `celeba_1n_eval.py`

Runs the large-gallery CelebA 1:N experiment with exact FAISS search. It evaluates gallery sizes of approximately 1,000, 2,500, 5,000, 7,500 and 9,000 identities using mated and non-mated probes.

A mated probe belongs to an identity already present in the gallery. A non-mated probe belongs to an identity absent from the gallery.

### `celeba_1n_fpir_recalibration.py`

Performs search-level threshold calibration using FPIR rather than only pairwise FMR. It reuses cached embeddings and does not require the face model to be rerun.

This experiment is diagnostic: it demonstrates that a threshold calibrated at one gallery size does not necessarily preserve the same search-level FPIR as the gallery grows.

### `celeba_rethreshold_existing_results.py`

Applies candidate thresholds to previously saved CelebA nearest-neighbour results. This avoids repeating embedding extraction and FAISS search when only the decision threshold changes.

### `pgu_beard_raccoon.py`

Evaluates clean-shaven versus beard/stubble appearance change using PGU-Face. A duplicate is counted as correctly detected only when the true subject is the nearest gallery identity and the nearest-neighbour distance satisfies the selected threshold.

The script also audits the dataset structure before inference.

## Key results

### CelebA large-gallery 1:N identification

The unconstrained single-image experiment produced the following Rank-1 identification rates:

| Actual gallery size | Rank-1 identification |
|---:|---:|
| 998 | 91.98% |
| 2,497 | 92.27% |
| 4,996 | 91.52% |
| 7,494 | 91.19% |
| 8,993 | 91.17% |

Rank-1 therefore remained approximately stable over the evaluated gallery range.

At the historical reference threshold `tau_l2 = 1.15`, search-level FPIR increased with gallery size. This shows why pairwise false-match performance should not be interpreted directly as the false-positive identification rate of a growing 1:N gallery.

The voting architecture therefore treats a biometric hit as a de-duplication alert to be reconciled with the voter-held credential, electoral-roll record and election-scoped voter reference, rather than as an autonomous ground for rejecting a voter.

### Search-level FPIR calibration

A repeated 1:N calibration experiment used a gallery size of 800 identities over 25 runs. At the nominal FPIR target of 0.3%, the selected threshold was approximately:

```text
tau_l2 = 1.15294
tau_sq = 1.32927
```

The observed calibration FPIR was approximately 0.38%, with calibration FNIR of approximately 8.72%.

When the same fixed threshold was applied to larger held-out galleries, FPIR increased as gallery size grew. This experiment is retained as evidence that search-level operating characteristics depend on gallery size; the manuscript does not treat this fixed threshold as a universal election-wide decision boundary.

### PGU-Face beard/stubble robustness

The corrected dataset audit recognised:

```text
subjects = 224
mapped images = 896
complete four-condition subjects = 224
unexpected image files = 0
duplicate mappings = 0
```

For the clean/beard experiment, 209 subjects had usable embeddings in both required conditions.

| Direction | Queries | Rank-1 | Thresholded duplicate detection at tau_l2=1.258 | FNIR |
|---|---:|---:|---:|---:|
| Clean -> beard | 209 | 99.04% | 99.04% | 0.96% |
| Beard -> clean | 209 | 99.04% | 99.04% | 0.96% |

At the earlier reference threshold `tau_l2 = 1.15`, Rank-1 remained 99.04%, while thresholded duplicate detection was 98.56%.

The PGU result is interpreted narrowly: ordinary beard/stubble appearance change did not materially reduce nearest-neighbour identification among successfully acquired faces. It is not an evaluation of prosthetic disguises, masks or presentation attacks.

## Environment setup

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r .\requirements.txt
```

If using the provided setup helper:

```powershell
.\setup_windows_py314.ps1
```

Check the environment before running experiments:

```powershell
python .\check_environment.py
```

The scripts should report the actual ONNX Runtime execution providers and model configuration used for each run.

## Data layout

A typical local layout is:

```text
data/
├── celeba/
│   ├── identity_CelebA.txt
│   └── img_align_celeba/
└── pgu/
    ├── 1/
    ├── 2/
    └── ...
```

The PGU root should contain numeric subject directories directly beneath `data/pgu/`.

## Result artefacts retained in Git

The repository should retain compact files needed to audit or reproduce manuscript tables, including:

- calibration summaries and threshold sweeps;
- CelebA 1:N summary tables and run configuration;
- search-level FPIR threshold candidates and summary;
- corrected PGU summary, validation/audit files, runtime environment and run configuration.

Large per-query CSV files are reproducible from the scripts and are not required in normal Git history. They may be archived separately as supplementary artefacts if needed.

## Files intentionally excluded

The following are not committed:

```text
.venv/
cache/
data/
__pycache__/
*.pyc
*.zip
```

Embedding caches are excluded because they are derived biometric templates and can be regenerated from the source datasets. Third-party face datasets are also excluded and must be obtained under their original licences.

Large per-query output files are excluded from routine Git history, particularly:

```text
results/celeba_1n_raccoon_l/celeba_1n_query_details.csv
results/calibration_1n_fpir_raccoon_l/celeba_1n_query_details_rethresholded.csv
results/calibration_1n_fpir_raccoon_l/mated_search_results_calibration.csv
results/calibration_1n_fpir_raccoon_l/nonmated_nearest_distance_l2.csv
```

## Reproducibility notes

The committed `celeba_identity_split.json` fixes the calibration/evaluation identity split used in the reported CelebA experiments.

Result directories include runtime and configuration JSON files where available. These should be retained with the corresponding aggregate CSVs so that model, threshold and environment settings can be traced.

Do not mix embedding caches generated by different InsightFace model packs.

## Interpretation

The face-search component is not intended to make an autonomous voter-eligibility decision. In the proposed voting workflow:

```text
voter credential / electoral-roll validation
        +
liveness
        +
1:N biometric de-duplication
        +
prior election-day voting record
        ↓
normal processing or supervised adjudication
```

A biometric candidate match is therefore an anomaly/de-duplication signal. Documentary and election-scoped identity evidence remain part of the decision process.

## Research context

These experiments support the AI component of the PhD research on secure, scalable and verifiable internet voting for India. They are intended to provide reproducible evidence for face de-duplication behaviour, not to claim demographic representativeness of CelebA or PGU-Face for the Indian electorate.
