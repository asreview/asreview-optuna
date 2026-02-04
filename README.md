# asreview-optuna

This repository provides tools for testing and optimizing the models found in ASReview via the Optuna package. It includes a Makita template that will generate a folder infrastructure with a jobs file to run the optimizations for a specific model.

## Installation

To get started, clone this repository:

```bash
git clone https://github.com/asreview/asreview-optuna.git
```

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Run Local
Simply execute the `main.py` file:
```bash
python ./src/main.py
```

Or check out all options:
```bash
python ./src/main.py -h
```

And, to see the results, start up the dashboard:
```bash
optuna-dashboard sqlite:///src/db.sqlite3
```

## Run on SLURM cluster
Two options here:
- A hosted, centralized DB
- A local DB

### Exoscale Hosted DB
#### To setup a DB
1. Create a PostgreSQL DB
2. Get the full URI using [exo cli](https://community.exoscale.com/documentation/tools/exoscale-command-line-interface/#installation) on a local machine `exo dbaas -z [DB ZONE] show [DB NAME] --uri`
3. Add the IP addresses from your study and dashboard servers to the IP filter

#### To run Optuna
```bash
sbatch --export=DB_URI="[YOUR DB_URI GOES HERE]" ./src/run_opruna.sh
```

### Local DB
```bash
sbatch ./src/run_optuna.sh
```

## Run on Exoscale
Two options here:
- A hosted, centralized DB
- A local DB

### Exoscale Hosted DB
#### To setup a DB
1. Create a PostgreSQL DB
2. Get the full URI using [exo cli](https://community.exoscale.com/documentation/tools/exoscale-command-line-interface/#installation) on a local machine `exo dbaas -z [DB ZONE] show [DB NAME] --uri`
3. Add the IP addresses from your study and dashboard servers to the IP filter

#### To start optuna-dashboard docker
1. Create a new instance (e.g., Ubuntu 24.04 Standard->Small, 50GIB)
    - Make sure to set your own SSH key
2. Update and reboot `sudo apt update && sudo apt upgrade` and `sudo reboot`
3. Install docker using the [official docker install instructions](https://docs.docker.com/engine/install/ubuntu/)
4. Check installation: `docker compose version`
5. Create dir and move into it `mkdir optuna-dashboard && cd optuna-dashboard/`
6. Create nginx.conf `nano nginx.conf` (example in `dashboard/nginx.conf`)
7. Install deps `sudo apt install -y apache2-utils`
8. Create htpasswd file `sudo htpasswd -c ./htpasswd admin`
9. Create docker-compose.yml `nano docker-compose.yml` (example in `deployment/dashboard/docker-compose.yml`, make sure to fill in the DB URI)
10. Start docker using docker-compose.yml `docker-compose up -d`
    
#### To Start a Study
1. Create a new instance (e.g., Ubuntu 24.04 CPU->Mega, 50GIB)
    - Make sure to set your own SSH key
    - Make sure to set the `asreview-and-optuna-dashboard` security group
2. Update and reboot `sudo apt update && sudo apt upgrade` and `sudo reboot`
3. Clone this repo `git clone https://github.com/asreview/asreview-optuna.git`
4. Move into dir `cd asreview-optuna`
5. Pull and checkout the correct study branch `git pull && git checkout [BRANCH_NAME]`
6. Install venv `sudo apt install python3.12-venv`
7. Create Python venv `python3 -m venv .venv`
8. Activate venv `source .venv/bin/activate`
9. Install Python packages `pip3 install -r requirements.txt`
10. Create dataset pickles `python3 ./src/feature_matrix_scripts/tfidf.py` (± 1.5 minutes)
11. Set `DB_URI` environment variable `export DB_URI=[FULL DB URI]`
12. Create a tmux environment so optuna keeps running when we close the connection `tmux new -s optuna`
    In the `optuna` tmux env run the following commands to start the study:
    1. `source .venv/bin/activate`
    2. `python3 src/main.py`
    3. Detach from the tmux environment using `ctrl` `+` `b` followed by `d` (you can always reattach using `tmux attach -t optuna`)
13. You are all set! Check the dashboard on your local machine through a browser: `[Exoscale instance ip]:8080`
14. You can see CPU usage using `htop`

### Local DB
1. Create a new instance (e.g., Ubuntu 24.04 CPU->Mega, 50GIB)
    - Make sure to set your own SSH key
    - Make sure to set the `asreview-and-optuna-dashboard` security group
2. Update and reboot `sudo apt update && sudo apt upgrade` and `sudo reboot`
3. Clone this repo `git clone https://github.com/asreview/asreview-optuna.git`
4. Move into dir `cd asreview-optuna`
5. Install venv `sudo apt install python3.12-venv`
6. Create Python venv `python3 -m venv .venv`
7. Activate venv `source .venv/bin/activate`
8. Install Python packages `pip3 install -r requirements.txt`
9. Create dataset pickles `python3 ./src/feature_matrix_scripts/tfidf.py` (± 1.5 minutes)
10. Set your simulation parameters in `main.py` using a cli editor such as `nano main.py`
11. Create a tmux environment so optuna keeps running when we close the connection `tmux new -s optuna`
    In the `optuna` tmux env run the following commands to start the study:
    1. `source .venv/bin/activate`
    2. `python3 src/main.py`
    3. Detach from the tmux environment using `ctrl` `+` `b` followed by `d` (you can always reattach using `tmux attach -t optuna`)
12. Create a tmux environment for the dashboard `tmux new -s dashboard`
    In the `dashboard` tmux env run the following commands to start the dashboard:
    1. `optuna-dashboard sqlite:///src/db.sqlite3 --host 0.0.0.0`
    2. Detach from the tmux environment using `ctrl` `+` `b` followed by `d` (you can always reattach using `tmux attach -t dashboard`)
13. You are all set! Check the dashboard on your local machine through a browser: `[Exoscale instance ip]:8080`
14. You can see CPU usage using `htop`

## Variables

### Study variables
- `VERSION` = Version number, reflected in the studyname
- `METRIC` = The metric used for optimization. Options:
    - `"loss"`: Loss
    - `"ndcg"`: Gain
- `STUDY_SET` = The combinations priors for synergy datasets. Options:
    - `"demo"`: 2 prior combinations per synergy dataset (14 * 2 simulations)
    - `"full"`: 10 prior combinations per synergy dataset (24 * 10 simulations)
- `CLASSIFIER_TYPE` = The ASReview2 classifier to use. Options: 
    - `"nb"`: Naive-Bayes
    - `"log"`: Logistic Classifier
    - `"svm"`: SVM
    - `"rf"`: Random Forest
- `FEATURE_EXTRACTOR_TYPE` = The ASReview2 feature extractor to use. Options:
    - `"tfidf"`: tfidf
    - `"onehot"`: onehot
- `PICKLE_FOLDER_PATH` = Path to optional preprocessed feature matrices that can be created using `optuna/feature_matrix_scripts/tfidf.py`
- `PRE_PROCESSED_FMS` = Flag to decide whether or not to use these preprocessed FMs, or to generate them on the fly.
- `PARALLELIZE_OBJECTIVE` = Flag to decide whether to parallelize the objective function
- `AUTO_SHUTDOWN` = Flag to decide whether or not to shut down after finishing a study (useful for exoscale)

### Optuna variables
- `OPTUNA_N_TRIALS` = Number of trials Optuna should run
- `OPTUNA_TIMEOUT` = Time in seconds, after which the current trial is cleanly finished and the study is wrapped up
- `OPTUNA_N_JOBS` = Number of Optuna trials to run in parallel (currently decided by `PARALLELIZE_OBJECTIVE`)

### Early stopping condition variables
- `MIN_TRIALS` = Number of trials before the stopping condition will be checked
    - If `curr_trial` >= `MIN_TRIALS` -> check stopping condition
- `N_HISTORY` = How far should the stopping condition look back?
- `STOPPING_THRESHOLD` = Threshold for checking whether to stop the study or not

# Questions and Contributions

If you have any questions or would like to contribute, please open an issue in the repository's [issues section](https://github.com/asreview/asreview-optuna/issues).

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Authors

The ASReview team.

This extension is part of the ASReview project (asreview.ai). It is maintained by the maintainers of ASReview LAB. See ASReview LAB for contact information and more resources.
