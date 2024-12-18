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