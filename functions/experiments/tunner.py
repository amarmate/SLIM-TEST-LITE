import pickle
import time
import pandas as pd
import mlflow
from pathlib import Path

from skopt import gp_minimize
from functions.metrics_test import *

class Tuner:
    def __init__(self, config, split_id,
                 gen_params, objective_fn,
                 mask):
        """
        Initialize the Tuner with configuration and parameters.
        
        Args:
            config (dict): Configuration dictionary containing experiment settings.
            split_id (int): Identifier for the data split.
            gen_params (dict): Parameters for the genetic algorithm including dataset and model settings.
            objective_fn (callable): Function to evaluate the performance of the model.
        """
        
        self.split_id = split_id
        self.gen_params = gen_params.copy()
        self.objective_fn = objective_fn
        self.space = config['SPACE_PARAMETERS']
        self.n_calls = config['N_SEARCHES_HYPER']
        self.n_start = config['N_RANDOM_STARTS']
        self.n_folds = config['N_CV']
        self.noise = config['NOISE_SKOPT']
        self.seed = config['SEED']
        self.name = gen_params['dataset_name']
        self.selector = gen_params['selector']
        self.PI = config['PI']
        self.suffix = config['SUFFIX_SAVE'] 

        self.calls_count = 0
        self.trial_results = []
        self.param_names = [dim.name for dim in self.space]

        self.gen_params.pop('X_test', None)
        self.gen_params.pop('y_test', None)

        self.save_dir = Path("..") / config['DATA_DIR'] / config['EXPERIMENT_NAME'] / config['TUNE_DIR'] / self.name / self.selector
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _wrapped_objective(self, params):
        p = dict(zip(self.param_names, params))
        gp_params = self.gen_params.copy()
        gp_params['log_level'] = 0

        if 'pop_iter_setting' in p:
            pis = int(p.pop('pop_iter_setting'))
            p['n_iter'], p['pop_size'] = self.PI[pis]
        
        gp_params.update(p)
        gp_params['it_tolerance'] = int(gp_params['it_tolerance'] * gp_params['n_iter'])
        self.temp_params = gp_params

        t0 = time.time()
        mean_rmse, stats = self.objective_fn(gp_params, self.split_id, n_splits=self.n_folds)
        elapsed = time.time() - t0

        self.calls_count += 1
        mlflow.set_tag("tuning_step", self.calls_count)

        record = {
            'trial_id': self.calls_count,
            'split_id': self.split_id,
            'seed': self.seed,
            **{k: gp_params[k] for k in p.keys()},
            'mean_rmse': mean_rmse,
            'elapsed_sec': elapsed,
            **stats,
        }

        mlflow.log_metric("tuning_time_sec", elapsed, step=self.calls_count)
        mlflow.log_metric("tuning_rmse", mean_rmse, step=self.calls_count)
        if 'mean_nodes' in stats:
            mlflow.log_metric("tuning_nodes", stats['mean_nodes'], step=self.calls_count)

        self.trial_results.append(record)
        return mean_rmse

    def tune(self):
        """
        Perform hyperparameter tuning using Bayesian optimization with Gaussian processes.
        Returns:
            dict: Best hyperparameters found during tuning with the dataset information included.
        """

        ckpt_dir = self.save_dir / f"checkpoint_tunning_split{self.split_id}_{self.suffix}.parquet"
        ckpt_params = self.save_dir / f"checkpoint_params_split{self.split_id}_{self.suffix}.pkl"

        if ckpt_params.exists():
            print(f"Checkpoint already exists: {ckpt_params}")
            with open(ckpt_params, 'rb') as f:
                best_params = pickle.load(f)
            return best_params

        with mlflow.start_run(run_name=f"train"):
            res = gp_minimize(
                func=self._wrapped_objective,
                dimensions=self.space,
                n_calls=self.n_calls,
                noise=self.noise,
                n_initial_points=self.n_start,
                random_state=self.seed,
                verbose=False
            )

            df = pd.DataFrame(self.trial_results)
            best_idx = df['mean_rmse'].idxmin()

            test_params = self.temp_params.copy()
            best_params = {}
            for key in test_params:
                if key in df.columns:
                    test_params[key] = df.loc[best_idx, key]
                    best_params[key] = test_params[key]

            best_score = float(df.loc[best_idx, 'mean_rmse'])

            mlflow.log_params(best_params)
            mlflow.log_metric("best_rmse", best_score)
            mlflow.set_tag("tunning_step", 'completed')

            test_params['bcv_rmse'] = best_score
            test_params.pop('X_train')
            test_params.pop('y_train')  

            df.to_parquet(ckpt_dir, index=False)
            with open(ckpt_params, 'wb') as f:
                pickle.dump(test_params, f)            
            return test_params 