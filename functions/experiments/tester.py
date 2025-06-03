import time
import pickle
from pathlib import Path

import pandas as pd
import mlflow

from functions.utils_test import pf_rmse_comp_time, log_latex_as_image


class Tester:
    def __init__(self, config, split_id, 
                 best_params, test_fn,
                 dataset, **kwargs):
        """
        Initalize the Tester class for running tests on a dataset with given parameters.
        Args:
            config (dict): Experiment-Konfiguration (N_TESTS, DATA_DIR, TEST_DIR, SEED, ...)
            split_id (int)
            best_params (dict): Hyperparameter-Dict vom Tuner
            test_fn (callable): Funktion test_fn(params, data_split, seed, **kwargs)
                                sie liefert (records_list, pop_stats_list, logs_list)
        """

        self.config = config
        self.split_id = split_id
        self.params = best_params.copy()
        self.dataset = dataset.copy()
        self.test_fn = test_fn

        self.name = best_params['dataset_name']
        self.selector = best_params['selector']

        self.N_TESTS = config['N_TESTS']
        self.seed = config['SEED']
        self.suffix = config['SUFFIX_SAVE']

        self.params.update(
            {
             'test_elite'   : True, 
             'log_level'    : 'evaluate',
             'it_tolerance' : 1e10,  # Remove tolerance for testing 
            }
        )

        self.save_dir = Path('..') / config['DATA_DIR'] / config['EXPERIMENT_NAME'] / config['TEST_DIR'] / self.name / self.selector
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        ckpt_test = self.save_dir / f"checkpoint_testing_split{self.split_id}_{self.suffix}.parquet"
        ckpt_pf = self.save_dir / f"checkpoint_pf_split{self.split_id}_{self.suffix}.pkl"
        ckpt_logs = self.save_dir / f"checkpoint_logs_split{self.split_id}_{self.suffix}.pkl"

        if ckpt_test.exists():
            print(f"Warning: Checkpoint already exists: {ckpt_test}, wasn't caught by the task manager.")

        all_records = []
        all_pop_stats = []
        all_logs = []

        try: 
            with mlflow.start_run(run_name=f"test"):
                mlflow.set_tag("testing_start", True)

                for test_n in range(self.N_TESTS):
                    mlflow.set_tag("testing_step", test_n+1)

                    records, pop_stats, logs = self.test_fn(
                        best_params = self.params, 
                        dataset = self.dataset, 
                        split_it = self.split_id,
                        seed= self.seed + test_n,
                    )

                    all_records.append(records)
                    all_pop_stats.extend(pop_stats)
                    all_logs.append(logs)

                    step = test_n+1
                    for metric, val in records.items():
                        if metric in ('rmse_test', 'mae_test', 'r2_test', 'nodes', 'time', 'gen_gap_per', 
                                      'overfit_per', 'ensemble_gap_per', 'nodes_count'):
                            mlflow.log_metric(f"testing_{metric}", val, step=step)

                df = pd.DataFrame(all_records)
                best_idx = df['rmse_test'].idxmin()
                # best_latex = df.loc[best_idx, 'latex_repr']
                # log_latex_as_image(best_latex, self.name, self.split_id,
                #                 prefix=self.config['PREFIX_SAVE'], 
                #                 suffix=self.config['SUFFIX_SAVE'],
                # )

                pf = pf_rmse_comp_time(all_pop_stats)
                mlflow.set_tag("testing_complete", True)

                df.to_parquet(ckpt_test, index=False)        

                with open(ckpt_pf, 'wb') as f:
                    pickle.dump(pf, f)
                with open(ckpt_logs, 'wb') as f:
                    pickle.dump(all_logs, f)
                mlflow.end_run(status="FINISHED")
            return df, pf, all_logs

        except Exception:
            mlflow.end_run(status="FAILED")
            raise 