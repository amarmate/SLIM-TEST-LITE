import numpy as np 
from functions.experiments.GP.config_gp import *
from functions.experiments.tracking import get_tasks

from functions.experiments.tunner import Tuner
from functions.experiments.GP.tune_gp import gp_tune

from functions.experiments.tester import Tester 
from functions.experiments.GP.test_gp import gp_test
from functions.experiments.github import commit_and_push

from joblib import Parallel, delayed, parallel_config



def run_experiment(config, task):
    tuner = Tuner(config=config, 
                objective_fn = gp_tune,
                **task)
    bp = tuner.tune()

    tester = Tester(config=config, 
                    test_fn=gp_test,
                    best_params=bp, 
                    **task)

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