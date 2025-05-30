import time
import pickle
from pathlib import Path

import pandas as pd
import mlflow

class Tester:
    def __init__(self, config, split_id, 
                 best_params, test_fn, 
                 mask):
        """
        Initalize the Tester class for running tests on a dataset with given parameters.
        Args:
            config (dict): Experiment-Konfiguration (N_TESTS, DATA_DIR, TEST_DIR, SEED, ...)
            name (str): Name des Datensatzes
            split_id (int)
            best_params (dict): Hyperparameter-Dict vom Tuner
            test_fn (callable): Funktion test_fn(params, data_split, seed, **kwargs)
                                sie liefert (records_list, pop_stats_list, logs_list)
            mask (list): boolean list of lists that indicates to which class the target belongs to. 
        """

        self.config = config
        self.split_id = split_id
        self.params = best_params
        self.test_fn = test_fn

        self.name = best_params['dataset_name']
        self.selector = best_params['selector']

        self.N_TESTS = config['N_TESTS']
        self.seed = config['SEED']
        self.suffix = config['SUFFIX_SAVE']

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

        with mlflow.start_run(run_name=f"{self.name}_split{self.split_id}_{self.selector}_test"):
            mlflow.set_tag("testing_start", True)

            for test_n in range(self.N_TESTS):
                mlflow.set_tag("testing_step", test_n+1)

                records, pop_stats, logs = self.test_fn(
                    self.params, self.data_split,
                    seed=self.seed + test_n,
                    selector=self.selector
                )
                # sammle alle Infos
                all_records.extend(records)
                all_pop_stats.extend(pop_stats)
                all_logs.append(logs)

                # Logge Metriken aus records[-1]
                last = records[-1]
                step = test_n+1
                for metric, val in last.items():
                    if metric in ('rmse_test','mae_test','r2_test','nodes','time_sec','gen_gap_%','overfit_%'):
                        mlflow.log_metric(f"testing_{metric}", val, step=step)

            # Abschließende Auswertung
            df = pd.DataFrame(all_records)
            best_idx = df['rmse_test'].idxmin()
            best_latex = df.loc[best_idx, 'latex_repr']
            # Default-Funktion, um LaTeX-Baum zu visualisieren
            log_latex_as_image(best_latex, self.name, self.split_id,
                               prefix=self.selector)

            pf = pf_rmse_comp_time(all_pop_stats)
            mlflow.set_tag("testing_complete", True)

            # Artefakte speichern und loggen
            df.to_pickle(ckpt_test)
            mlflow.log_artifact(str(ckpt_test))
            with open(ckpt_pf, 'wb') as f:
                pickle.dump(pf, f)
            mlflow.log_artifact(str(ckpt_pf))
            with open(ckpt_logs, 'wb') as f:
                pickle.dump(all_logs, f)
            mlflow.log_artifact(str(ckpt_logs))

        return df, pf, all_logs
