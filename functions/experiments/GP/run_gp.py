import numpy as np 
import time 
from functions.experiments.GP.config_gp import *
from functions.experiments.tracking import get_tasks

from functions.experiments.tunner import Tuner
from functions.experiments.GP.tune_gp import gp_tune

from functions.experiments.tester import Tester 
from functions.experiments.GP.test_gp import gp_test
from functions.experiments.github import periodic_commit

from joblib import Parallel, delayed, parallel_config
import threading



def run_experiment(config, task):
    print(f"Running task: {task['name']} with selector {task['selector']} and split {task['split_id']}")
    tuner = Tuner(config=config, 
                objective_fn = gp_tune,
                **task)
    bp = tuner.tune()


    print(f'Tunning completed for {task["name"]} with selector {task["selector"]} and split {task["split_id"]}. Best parameters: {bp}')
    print(f"Running testing for task: {task['name']} with selector {task['selector']} and split {task['split_id']}")

    tester = Tester(config=config, 
                    test_fn=gp_test,
                    best_params=bp, 
                    **task)
    tester.run()

    print(f'Testing completed for {task["name"]} with selector {task["selector"]} and split {task["split_id"]}. Results saved.')

def run_gp(args):
    np.random.seed(SEED)

    commit_thread = threading.Thread(
        target=periodic_commit, args=(config), daemon=True
    )
    commit_thread.start()

    tasks = get_tasks(args, config)

    with parallel_config(n_jobs=args.workers, prefer='threads', verbose=10):
        Parallel()(delayed(run_experiment)(config, task) for task in tasks)
