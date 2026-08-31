# asreview-optuna

This repository provides tools for testing and optimizing the models found in ASReview via the Optuna package. It includes a Makita template that will generate a folder infrastructure with a jobs file to run the optimizations for a specific model.

# Installation

To get started, clone this repository:

```bash
git clone https://github.com/asreview/asreview-optuna.git
```

Make sure you have all dependencies installed:

```bash
uv sync
```

# Data

This project tunes against the local [synergy_plus](https://github.com/asreview/synergy-dataset) mirror: one CSV per systematic-review dataset, plus `metadata/review_metadata.csv` which assigns each dataset to a `train` or `test` split (a whole dataset belongs to exactly one split). `--data-path` (required on every script below) must point at this directory.

Before running anything else, generate the random-prior study files once:
```bash
python ./src/generate_studies.py --data-path /path/to/synergy_plus
```
This writes `synergy_studies_{train,test,demo}.jsonl` into `src/studies/`. Re-run it (optionally with `--n-priors`/`--seed`) to regenerate with a different repeat count or seed.

To test whether different preconditions (known ahead of screening, or diagnostic post-hoc ones) call for different hyperparameters, add `--stratify-by`:
```bash
python ./src/generate_studies.py --data-path /path/to/synergy_plus --stratify-by domain search_size inclusion_ratio n_databases protocol
```
This additionally partitions the `train` split into one `synergy_studies_train-<axis>-<stratum>.jsonl` per stratum, plus `stratification_manifest.json` recording every dataset's stratum on every axis computed so far. Running `--stratify-by` for one axis and later for another (against the same `--studies-path`) merges into the same manifest rather than overwriting it, so it's safe to add axes incrementally. Available axes:

| Axis | Strata | Notes |
| --- | --- | --- |
| `domain` | `health`, `nonhealth` | Requires the extended metadata variant (`primary_topic_domain` column) — by default expected at `<data-path>_extended/metadata/review_metadata.csv`, overridable with `--extended-metadata-path`. |
| `search_size` | `small`, `medium`, `large` | Tertiles of `n_records` (total search size), computed from the train split. |
| `inclusion_ratio` | `low`, `mid`, `high` | Tertiles of `n_records_included / n_records`. Not knowable ahead of screening — a post-hoc diagnostic axis, not an actionable precondition. |
| `n_databases` | `low`, `mid`, `high` | Tertiles of `number_of_databases`. Since this is a clustered discrete integer (most reviews search 3-4 databases), the tertiles can come out uneven — check `stratification_manifest.json`'s `n_databases_axis.tertile_boundaries` before reading too much into strata sizes. |
| `protocol` | `protocol`, `no_protocol` | Direct split on the `protocol` (pre-registered) flag — no tertile computation. |
| `baseline_loss` | `low`, `mid`, `high` | Tertiles of a baseline study's per-dataset loss. Requires `--baseline-loss-path`, a CSV with `dataset_id`/`loss_mean` columns covering **every** dataset in both splits — produce the train-side half with `evaluate_test.py --study-set train --skip-baseline` (see below) and concatenate it with an existing `test_results_*.csv`'s `Tuned` rows for the test-side half. Also a post-hoc diagnostic axis, not an actionable precondition. |

If you want to tune with `--feature-extractor mxbai` or `--feature-extractor multilingual-e5` (precomputed embeddings, not tuned by Optuna, see the options table below), precompute them once:
```bash
sbatch --export=DATA_PATH="/path/to/synergy_plus" ./src/preprocess_fms.sh
```
or locally: `python ./src/feature_matrix_scripts/mxbai.py --data-path /path/to/synergy_plus` (and similarly for `multilingual_e5.py`). Both accept a repeatable `--dataset-id` to cheaply precompute just one dataset for smoke testing. `tfidf`/`onehot` need no offline step — they're tuned every trial, so `main.py` fits them on the fly (once per dataset per trial, not once per prior draw).

# Run Local
Simply execute the `main.py` file:
```bash
python ./src/main.py --data-path /path/to/synergy_plus
```
The cheapest way to sanity-check the whole pipeline is `--study-set demo` (1 dataset, 2 prior draws).

Or check out all options:
```bash
python ./src/main.py -h
```

Once a study finishes, evaluate its best hyperparameters against the held-out test split (never touched during the search):
```bash
python ./src/evaluate_test.py --study-name "[the name printed in main.py's banner]" --classifier svm --feature-extractor tfidf --data-path /path/to/synergy_plus
```
This prints and writes a per-dataset breakdown CSV. By default it also runs the relevant ASReview-shipped ELAS baseline model(s) against the same test split for comparison — `tfidf` against ELAS u3 and u4, `mxbai` against ELAS h3, `multilingual-e5` against ELAS l2 (`onehot` has no ELAS equivalent) — using their exact shipped hyperparameters (from `asreview.models.models`) rather than this repo's tuning defaults. Pass `--skip-baseline` to omit this.

`--study-set` (default `test`) can point this at any other `synergy_studies_*.jsonl` instead — e.g. `--study-set train --skip-baseline` evaluates a study's tuned hyperparameters against the (non-held-out) train split, which is how you produce the per-dataset loss CSV `generate_studies.py --stratify-by baseline_loss` needs.

And, to see the results of your optimization, start up the dashboard:
```bash
optuna-dashboard sqlite:///src/db.sqlite3
```

# Run on SLURM cluster
Two options here:
- A hosted, centralized DB
- A local DB

## Exoscale Hosted DB

Steps to setup an Exoscale DB:

1. Create a PostgreSQL DB
2. Get the full URI using [exo cli](https://community.exoscale.com/documentation/tools/exoscale-command-line-interface/#installation) on a local machine `exo dbaas -z [DB ZONE] show [DB NAME] --uri`
3. Add the IP addresses from your study and dashboard servers to the IP filter

Then, run ASReview Optuna:
```bash
sbatch --export=ALL,DATA_PATH="/path/to/synergy_plus",DB_URI="[YOUR DB_URI GOES HERE]" ./src/run_single.sh
```

## Local DB
When you choose to use a local db (sqlite), you can simply run:
```bash
sbatch --export=ALL,DATA_PATH="/path/to/synergy_plus" ./src/run_single.sh
```

To sweep multiple classifier/feature-extractor combinations, edit the constants block at the top of `run_single.sh` (`STUDY_SET`, `CLASSIFIER`, `FEATURE_EXTRACTOR`, `METRIC`, `N_TRIALS`) and submit again — each `sbatch` call is its own independent SLURM job with its own guaranteed resource allocation. Point every submission at the same `DB_URI` (a real client-server DB, not sqlite — see above) to run several combos concurrently: separate jobs get real per-job core isolation from the scheduler, which trying to background multiple runs inside one job's allocation can't guarantee, and a proper DB backend handles the resulting concurrent writes safely (sqlite does not, under concurrent writers).

`preprocess_fms.sh` (see [Data](#data) above) needs a GPU partition — its `--partition=gpu` placeholder likely needs adjusting to your cluster's actual GPU partition name.

Note the `ALL,` prefix on `--export`: without it, Slurm restricts the job's environment to *only* the variables listed, dropping `PATH` and breaking `module load`/venv activation.

## Stratified sweep
To tune per-stratum instead of over the whole `train` split (see [Data](#data) above for `--stratify-by`), use `hpc/run_stratum.sh` — a Slurm **array job**: `STUDY_SETS` is a hand-edited bash array listing every stratum's study-set name, and `#SBATCH --array=0-N` (N = number of entries minus one — Slurm parses `#SBATCH` lines before the script runs, so this can't be derived from the list automatically; keep the two in sync) launches one task per entry, each picking its own `STUDY_SET` via `$SLURM_ARRAY_TASK_ID`. `CLASSIFIER`/`FEATURE_EXTRACTOR` are fixed to `svm`/`tfidf` for fast iteration. One edit, one submission, regardless of how many strata are listed:
```bash
sbatch --export=ALL,DATA_PATH="/path/to/synergy_plus",DB_URI="[YOUR DB_URI GOES HERE]" hpc/run_stratum.sh
```
Each array task still gets its own independent resource allocation and log file (`logs/run_stratum_<jobid>_<taskindex>.out`/`.err`), same as separate `sbatch` calls would — Slurm just handles the queueing. Point every job at the same real `DB_URI` since they run concurrently. To retry or rerun just one stratum without resubmitting the whole array, override the range at submission time, e.g. `sbatch --array=3 ... hpc/run_stratum.sh`.

# ASReview Optuna Options
| Option                                                    | Description                                                                                                          |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `-h, --help`                                               | Show this help message and exit                                                                                     |
| `--metric {loss,ndcg}`                                     | The metric used as objective during optimization.                                                                   |
| `--study-set STUDY_SET`                                    | The study set to use: `demo`, `train`, or a stratum name produced by `generate_studies.py --stratify-by` (e.g. `train-domain-health`). Test-split data is intentionally not selectable here; use `evaluate_test.py`. |
| `--classifier {log,nb,rf,svm}`                              | The classifier to optimize.                                                                                        |
| `--feature-extractor {tfidf,onehot,mxbai,multilingual-e5}`  | The feature extractor to optimize. `mxbai`/`multilingual-e5` have no on-the-fly implementation and require `--pre-processed-fms`. |
| `--balancer {ratio,double}`                                 | The balancer to optimize (default: `ratio`).                                                                       |
| `--pre-processed-fms`                                       | If set, use the pre-processed feature matrices.                                                                    |
| `--n-trials N_TRIALS`                                       | Set the maximum number of trials that will be ran.                                                                 |
| `--parallelize-objective`                                   | If set, run one trial with several processes. Each process will run 1 study set row at a time.                    |
| `--n-workers N_WORKERS`                                     | Set the number of workers used for parallelizing the objective.                                                    |
| `--data-path DATA_PATH`                                     | The path to the synergy_plus data directory (required).                                                            |
| `--fms-path FMS_PATH`                                       | The path to the preprocessed feature matrices (default: `src/preprocessed_fms`).                                   |
| `--studies-path STUDIES_PATH`                               | The path to the studies JSON files, `demo` and `train` (default: `src/studies`).                                   |
| `--seed SEED`                                               | Seed for the Optuna sampler, for a reproducible hyperparameter search order (default: `42`).                       |

# Questions and Contributions

If you have any questions or would like to contribute, please open an issue in the repository's [issues section](https://github.com/asreview/asreview-optuna/issues).

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Authors

The ASReview team.

This extension is part of the ASReview project (asreview.ai). It is maintained by the maintainers of ASReview LAB. See ASReview LAB for contact information and more resources.
