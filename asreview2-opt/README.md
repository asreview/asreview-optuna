# ASReview 2.x NB TfIDF Hyperparameter Optimization
## Install and Run
1. Install dependencies `pip3 install -r requirements.txt`
2. Run `feature_matrices.py` (± 40 seconds)
3. Run `main.py` to execute the Optuna trials
4. Run `optuna-dashboard sqlite:///db.sqlite3` to see the Optuna dashboard

## Variables
### Study variables
- `PICKLE_FOLDER_PATH` = Path to feature matrices that were created by `feature_matrices.py`
- `N_STUDIES` = Number of studies to take from the list of 1000 in `synergy_studies_1000.jsonl`
- `CLASSIFIER_TYPE` = The ASReview2 classifier to use. Options: 
    - `"nb"`: Naive-Bayes
    - `"log"`: Logistic Classifier
    - `"svm"`: SVM
    - `"rf"`: Random Forest
- `STUDY_NAME` = The name of the study, used to differentiate between studies in the sqlite DB and the optuna dashboard
- `PARALLELIZE_OBJECTIVE` = 
    - `True`: Parallelize the objective function across all (-2) available CPU cores (asreview simulations in parallel)
    - `False`: Parallelize trials across all (-2) CPU cores (Optuna trials in parallel)

### Optuna variables
- `OPTUNA_N_TRIALS` = Number of trials Optuna should run
- `OPTUNA_TIMEOUT` = Time in seconds, after which the current trial is cleanly finished and the study is wrapped up
- `OPTUNA_N_JOBS` = Number of Optuna trials to run in parallel (currently decided by `PARALLELIZE_OBJECTIVE`)

### Early stopping condition variables
- `MIN_TRIALS` = Number of trials before the stopping condition will be checked
    - If `curr_trial` >= `MIN_TRIALS` -> check stopping condition
- `N_HISTORY` = How far should the stopping condition look back?
- `STOPPING_THRESHOLD` = Threshold for checking whether to stop the study or not

## Exoscale Instructions
1. Create a new instance (e.g., Ubuntu 24.04 CPU->Mega, 50GIB)
    - Make sure to set your own SSH key
    - Make sure to set the `asreview-and-optuna-dashboard` security group
2. Update and reboot `sudo apt update && sudo apt upgrade` and `sudo reboot`
3. Clone this repo `git clone https://github.com/asreview/asreview-optuna.git`
4. Move into dir `cd asreview-optuna/asreview2-opt/`
5. Install venv `sudo apt install python3.12-venv`
6. Create Python venv `python3 -m venv .venv`
7. Activate venv `source .venv/bin/activate`
8. Install Python packages `pip3 install -r requirements.txt`
9. Create dataset pickles `python3 feature_matrices.py` (± 1.5 minutes)
10. Set your simulation parameters in `main.py` using a cli editor such as `nano main.py`
11. Create a tmux environment so optuna keeps running when we close the connection `tmux new -s optuna`
    In the `optuna` tmux env run the following commands to start the study:
    1. `source .venv/bin/activate`
    2. `python3 main.py`
    3. Deattach from the tmux environment using `ctrl` `+` `b` followed by `d` (you can always reattach using `tmux attach -t optuna`)
12. Create a tmux environment for the dashboard `tmux new -s dashboard`
    In the `dashboard` tmux env run the following commands to start the study:
    1. `optuna-dashboard sqlite:///db.sqlite3 --host 0.0.0.0`
    2. Deattach from the tmux environment using `ctrl` `+` `b` followed by `d` (you can always reattach using `tmux attach -t dashboard`)
13. You are all set! Check the dashboard on your local machine through a browser: `[Exoscale instance ip]:8080`
14. You can see CPU usage using `htop`
