from functions.experiments.MULTIGP.run_multi import run_multi
from functions.experiments.MULTIGP.config_multi import config

from functions.experiments.parse import parse_args
from functions.experiments.github import init_or_update_repo
from functions.experiments.mlflow import cleanup_running_runs

if __name__ == "__main__":
    try: 
        args = parse_args(config)

        # init_or_update_repo(config)

        cleanup_running_runs()
        
        run_multi(args)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
