from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import os
os.environ['MLFLOW_TRACKING_URI'] = 'file:../data/mlruns'

def cleanup_running_runs():
    print("Cleaning up RUNNING and FAILED runs...")
    client = MlflowClient()

    experiments = client.search_experiments(view_type=ViewType.ALL)

    for exp in experiments:
        if exp.experiment_id == '0':
            continue

        runs = client.search_runs(
            experiment_ids=exp.experiment_id,
            filter_string="attributes.status = 'FAILED'",
            max_results=100
        )

        runs += client.search_runs(
            experiment_ids=exp.experiment_id,
            filter_string="attributes.status = 'RUNNING'",
            max_results=100
        )

        for run in runs:
            print(f"Deleting RUNNING run: {run.info.run_id} from experiment '{exp.name}'")
            client.delete_run(run.info.run_id)