# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid
import os
import warnings

import numpy as np 

from slim_gsgp_lib_np.main_slim import slim
from slim_gsgp_lib_np.config.multi_slim_config import *

# -------------------------------- MULTI_SLIM --------------------------------
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.multi_slim import MULTI_SLIM

# -----------------------------------  GP -----------------------------------
from slim_gsgp_lib_np.algorithms.GP.gp import GP
from slim_gsgp_lib_np.algorithms.GP.operators.mutators import mutate_tree_subtree
from slim_gsgp_lib_np.algorithms.GP.representations.tree_utils import tree_depth
from slim_gsgp_lib_np.config.gp_config import *
from slim_gsgp_lib_np.selection.selection_algorithms import tournament_selection_max, tournament_selection_min
from slim_gsgp_lib_np.utils.logger import log_settings

ELITES = {}
UNIQUE_RUN_ID = uuid.uuid1()
ALGORITHM = "MULTI_SLIM"

def multi_slim(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None, dataset_name: str = None,
        slim_parameters: dict = None, slim_version: str = "SLIM+SIG2",
        pop_size : int = multi_pi_init["pop_size"], 
        seed: int = multi_params["seed"],
        verbose: int = multi_solve_params["verbose"],
        n_iter : int = multi_solve_params["n_iter"],
        log_level: int = multi_solve_params["log"],
        minimization: bool = True,
        log_path: str = None,


        callbacks: list = None, 
        timeout: int = 100,
        full_return: bool = False
):
    """
    Main function to execute the MULTI_SLIM GSGP algorithm on specified datasets.

    """
    # Validate the inputs given  TODO 
    # validate_parameters()

    # Calling the SLIM-GSGP algorithm 
    elite, population = slim(
        X_train=X_train, y_train=y_train, dataset_name=dataset_name, test_elite=False,
        full_return=True, seed=seed, verbose=verbose, log_level=log_level,
        run_info=[ALGORITHM, slim_version, UNIQUE_RUN_ID, dataset_name], minimization=minimization,
        log_path=log_path,
          **slim_parameters)
        
    # ------------------------ PI INIT ------------------------------
    multi_pi_init['FUNCTIONS'] = elite.collection[0].FUNCTIONS
    multi_pi_init['TERMINALS'] = elite.collection[0].TERMINALS
    multi_pi_init['CONSTANTS'] = elite.collection[0].CONSTANTS
    multi_pi_init['SPECIALISTS'] = {f'S_{i}' : ind for i, ind in enumerate(population.population)}

    multi_pi_init['p_constant'] = 0
    multi_pi_init['p_terminal'] = 0 
    multi_pi_init['p_specialist'] = 0 
    multi_pi_init['pop_size'] = pop_size
    multi_pi_init['depth_condition'] = 0   # The max depth of each of the conditions 
    multi_pi_init['max_depth'] = 0         # The max depth of the tree

    # ------------------- MULTI_SLIM PARAMETERS ----------------------


    # ---------------- MULTI_SLIM SOLVE PARAMETERS --------------------


    # --------------- Run the MULTI_SLIM algorithm --------------------
    optimizer = MULTI_SLIM(pi_init=multi_pi_init, **multi_params)
    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **multi_solve_params
    )

    log_settings(
        path=os.path.join(os.getcwd(), "log", "slim_settings.csv"),
        settings_dict=[slim_parameters,
                       multi_pi_init,
                       multi_params, 
                       multi_solve_params],
        unique_run_id=UNIQUE_RUN_ID
    ) if log_level > 0 else None

    optimizer.elite.version = slim_version
    optimizer.elite.iteration = optimizer.iteration
    optimizer.elite.early_stop = optimizer.stop_training
    
    if full_return: 
        return optimizer.elite, optimizer.population
    
    return optimizer.elite
