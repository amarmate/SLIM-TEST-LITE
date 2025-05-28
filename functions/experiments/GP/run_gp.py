import numpy as np 
from functions.experiments.GP.config_gp import *
from functions.experiments.tracking import get_tasks
from functions.experiments.GP.tune_gp import tuning
from functions.experiments.GP.test_gp import test_gp
from functions.experiments.github import commit_and_push

# Parallel libraries 
from joblib import Parallel, delayed, parallel_config



def run_experiment(config, task):
    # task: 
        # Returns:
        # list: A list of tasks to be executed, where each task is a tuple containing:
        #     - gen_params: A dictionary with parameters for the genetic algorithm.
        #     - split_id: The split identifier for the task.
        #     - mask: The mask for the task, if applicable.

    pass 





def run_gp(args):
    np.random.seed(SEED)
    
    # 1. Create the tasks to be executed
    tasks = get_tasks(args, config)

    # 2. Parallel execution of tuning and testing - ADD A TIMEOUT WHEN NO PROGRESS IS BEING MADE 
    with parallel_config(n_jobs=args.workers, prefer='threads', verbose=10):
        # Tuning phase
        Parallel()

    pass