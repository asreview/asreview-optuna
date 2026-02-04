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

# Run Local
Simply execute the `main.py` file:
```bash
python ./src/main.py
```

Or check out all options:
```bash
python ./src/main.py -h
```

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
sbatch --export=DB_URI="[YOUR DB_URI GOES HERE]" ./src/run_single.sh
```

## Local DB
When you choose to use a local db (sqlite), you can simply run:
```bash
sbatch ./src/run_single.sh
```

# ASReview Optuna Options
| Option                               | Description                                                                                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-h, --help`                         | Show this help message and exit                                                                                                                             |
| `--metric {loss,ndcg}`               | The metric used as objective during optimization.                                                                                                           |
| `--study-set {demo,full}`            | The study set that is used.                                                                                                                                 |
| `--classifier {log,nb,rf,svm}`       | The classifier to optimize.                                                                                                                                 |
| `--feature-extractor {tfidf,onehot}` | The feature extractor to optimize.                                                                                                                          |
| `--pre-processed-fms`                | If set, use the pre-processed feature matrices.                                                                                                             |
| `--n-trials N_TRIALS`                | Set the maximum number of trials that will be ran.                                                                                                          |
| `--parallelize-objective`            | If set, run one trial with several threads. Each thread will run 1 study set row at a time. Useful if you have a lot of studies (e.g., `study-set="full"`). |
| `--n-workers N_WORKERS`              | Set the number of workers used for parallelizing the objective.                                                                                             |
| `--data-path DATA_PATH`              | The path to the raw data.                                                                                                                                   |
| `--fms-path FMS_PATH`                | The path to the preprocessed feature matrices.                                                                                                              |
| `--studies-path STUDIES_PATH`        | The path to the studies JSON files (`demo` and `full`).                                                                                                     |

# Questions and Contributions

If you have any questions or would like to contribute, please open an issue in the repository's [issues section](https://github.com/asreview/asreview-optuna/issues).

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Authors

The ASReview team.

This extension is part of the ASReview project (asreview.ai). It is maintained by the maintainers of ASReview LAB. See ASReview LAB for contact information and more resources.
