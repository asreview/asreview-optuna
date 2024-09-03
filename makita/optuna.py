import optuna
import fire
import subprocess
from asreviewcontrib.insights.metrics import loss
from asreview import open_state

def start_optimization(output_folder: str, database_name: str, n_trials: int, classifier_name: str):

    # get all the datasets in the data folder (alternative collect them during makita run)
    datasets = None

    def objective(trial):
        loss_values = []

        for fp_dataset in datasets:
            loss_value = _run_simulation(trial, output_folder, classifier_name, fp_dataset)
            loss_values.append(loss_value)
        
        return loss_values

    study = optuna.create_study(storage=f"sqlite:///{database_name}", 
                                study_name=classifier_name,
                                direction=["minimize"] * len(datasets))
    study.optimize(objective, n_trials=n_trials)

    print(f"Optimization finished. Best value: {study.best_value}, Best parameters: {study.best_params}")

def _run_simulation(trial : int, output_folder : str, classifier : str, fp_dataset : str):

    command = [
        "python", "-m", "asreview", "simulate", fp_dataset, 
        "-s", f"{output_folder}/simulation/{dataset}/simulation_{trial}.asreview", 
        "-m", classifier
    ]

    subprocess.run(command, check=True)

    # get the loss from the simulation we just ran

    return loss

if __name__ == '__main__':
    fire.Fire(start_optimization)