import numpy as np 
import os 
import random
import mlflow 
from functions.experiments.MULTIGP.config_multi import *
from functions.experiments.tracking import get_tasks

from functions.experiments.tunner import Tuner
from functions.experiments.MULTIGP.tune_multi import multi_tune, config_1, config_2

from functions.experiments.tester import Tester 
from functions.experiments.MULTIGP.test_multi import multi_test
from functions.experiments.github import periodic_commit

from joblib import Parallel, delayed, parallel_config
import threading


def run_experiment(config, task):
    name, selector, split_id = task['gen_params']['dataset_name'], task['gen_params']['selector'], task['split_id']
    mlflow.set_experiment(f"{EXPERIMENT_NAME}_{name}_{selector}_split{split_id}")

    print(f"Running task: {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}")
    config1 = config_1(config)

    tuner_step1 = Tuner(config=config1, 
                        objective_fn = multi_tune,
                        **task)
    bp1 = tuner_step1.tune(run='multi1') 
    print(f'Tunning step 1 completed for {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}')

    print(f"Running step 2 for task: {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}")
    config2, task = config_2(config, bp1, task)

    tuner_step2 = Tuner(config=config2, 
                        objective_fn = multi_tune,
                        **task)
    params = tuner_step2.tune(run='multi2')
    print(f'Tunning step 2 completed for {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}')

    print(f'Testing for task: {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}')
    tester = Tester(config=config, 
                    test_fn=multi_test,
                    best_params=params, 
                    **task)
    tester.run()
    print(f'Testing completed for {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}')





# ------------------------------------------------- MAIN FUNCTION ------------------------------------------------- #
def run_multi(args):
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)



    tasks = get_tasks(args, config)

    mlflow.set_tracking_uri("file:../data/mlruns")

    # commit_thread = threading.Thread(
    #     target=periodic_commit, args=(config,), daemon=True
    # )
    # commit_thread.start()
    
    with parallel_config(n_jobs=args.workers, 
                         prefer='processes',
                         # verbose=10
                        ):
        Parallel()(delayed(run_experiment)(config, task) for task in tasks)
