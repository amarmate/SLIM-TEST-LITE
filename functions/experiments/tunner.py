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
                 step=0):
        """
        Initialize the GPTuner with configuration and parameters.
        
        Args:
            config (dict): Configuration dictionary containing experiment settings.
            split_id (int): Identifier for the data split.
            gen_params (dict): Parameters for the genetic algorithm including dataset and model settings.
            objective_fn (callable): Function to evaluate the performance of the model.
            step (int, optional): Step number for tuning. Defaults to 0. If using multi-slim, then the second step is 1.
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
        self.PI = config['PI_SETTINGS']
        self.step = step

        self.calls_count = 0
        self.trial_results = []
        self.param_names = [dim.name for dim in self.space]

        self.gen_params.pop('X_test', None)
        self.gen_params.pop('y_test', None)

        self.save_dir = Path("../") / config['DATA_DIR'] / config['TUNE_DIR'] / step / self.name / self.selector
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _wrapped_objective(self, params):
        p = dict(zip(self.param_names, params))
        gp_params = self.gen_params.copy()

        if 'pop_iter_setting' in p:
            pis = int(p.pop('pop_iter_setting'))
            gp_params['n_iter'], gp_params['pop_size'] = self.PI[pis]
        gp_params.update(p)

        t0 = time.time()
        mean_rmse, stats = self.objective_fn(gp_params, self.split_id, n_splits=self.n_folds)
        elapsed = time.time() - t0

        self.calls_count += 1
        mlflow.set_tag("tuning_step", self.calls_count)

        record = {
            'trial_id': self.calls_count,
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
        with mlflow.start_run(run_name=f"{self.name}_split{self.split_id}_{self.selector}"):
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
            best_params = df.loc[best_idx, self.param_names].to_dict()  
            best_score = float(df.loc[best_idx, 'mean_rmse'])

            mlflow.log_params(best_params)
            mlflow.log_metric("best_rmse", best_score)

            df.to_parquet(self.save_dir / f"checkpoint_tunning{self.split_id}.parquet", index=False)
            mlflow.log_artifact(str(self.save_dir / f"checkpoint_tunning{self.split_id}.parquet"))
            with open(self.temp_dir / f"checkpoint_params{self.split_id}.pkl", 'wb') as f:
                pickle.dump(best_params, f)

            return df, best_params, best_score, res

