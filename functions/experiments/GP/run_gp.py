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
    name, selector, split_id = task['gen_params']['dataset_name'], task['gen_params']['selector'], task['split_id']
    print(f"Running task: {name} / selector {selector} / split {split_id}")
    tuner = Tuner(config=config, 
                objective_fn = gp_tune,
                **task)
    params = tuner.tune()

    print(f'Tunning completed for {name} / selector {selector} / split {split_id}')
    print(f"Running testing for task: {name} / selector {selector} / split {split_id}")

    tester = Tester(config=config, 
                    test_fn=gp_test,
                    best_params=params, 
                    **task)
    tester.run()

    print(f'Testing completed for {name} / selector {selector} / split {split_id}')



def run_gp(args):
    np.random.seed(SEED)

    commit_thread = threading.Thread(
        target=periodic_commit, args=(config), daemon=True
    )
    commit_thread.start()

    tasks = get_tasks(args, config)

    with parallel_config(n_jobs=args.workers, prefer='threads', verbose=10):
        Parallel()(delayed(run_experiment)(config, task) for task in tasks)
