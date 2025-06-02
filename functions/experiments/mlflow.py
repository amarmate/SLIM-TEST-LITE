from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus

def cleanup_running_runs():
    client = MlflowClient()
    experiments = client.list_experiments()
    
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            max_results=1000
        )
        for run in runs:
            print(f"Deleting RUNNING run: {run.info.run_id} from experiment '{exp.name}'")
            client.delete_run(run.info.run_id)
