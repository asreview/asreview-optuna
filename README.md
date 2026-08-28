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

To test whether different preconditions (known ahead of screening) call for different hyperparameters, add `--stratify-by`:
```bash
python ./src/generate_studies.py --data-path /path/to/synergy_plus --stratify-by domain search_size
```
This additionally partitions the `train` split into `synergy_studies_train-domain-{health,nonhealth}.jsonl` and `synergy_studies_train-size-{small,medium,large}.jsonl` (size tertiles computed from the train split's `n_records`), plus `stratification_manifest.json` recording every dataset's stratum on every axis computed so far. `domain` requires the extended metadata variant (`primary_topic_domain` column) — by default expected at `<data-path>_extended/metadata/review_metadata.csv`, overridable with `--extended-metadata-path`. Running `--stratify-by` for one axis and later for another (against the same `--studies-path`) merges into the same manifest rather than overwriting it, so it's safe to add axes incrementally.

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
To tune per-stratum instead of over the whole `train` split (see [Data](#data) above for `--stratify-by`), use `hpc/run_stratum.sh` — same hand-edited-constants-block pattern as `run_tfidf.sh`, but with `STUDY_SET` set to a stratum name and `CLASSIFIER`/`FEATURE_EXTRACTOR` fixed to `svm`/`tfidf` for fast iteration. Edit `STUDY_SET` to each of `train-domain-health`, `train-domain-nonhealth`, `train-size-small`, `train-size-medium`, `train-size-large` in turn and submit after each edit:
```bash
sbatch --export=ALL,DATA_PATH="/path/to/synergy_plus",DB_URI="[YOUR DB_URI GOES HERE]" hpc/run_stratum.sh
```
As above, point every job at the same real `DB_URI` since they run concurrently.

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
