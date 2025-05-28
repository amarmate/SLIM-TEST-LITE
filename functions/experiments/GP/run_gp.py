import numpy as np 
from functions.experiments.GP.config_gp import *
from functions.experiments.GP.tracking import get_tasks
from functions.experiments.GP.tune_gp import tuning
from functions.experiments.GP.test_gp import test_gp
from functions.experiments.github import commit_and_push





def run_gp(args):
    np.random.seed(SEED)
    
    # 1. Create the tasks to be executed
    tasks = get_tasks(args)

    # 2. 

    # PARALLELIZATION - ADD A TIMEOUT SO THAT AFTER A MOMENT THE SYSTEMS RESETS (e.g. 1 hour)
    # 4. Timeout / Watchdog (komplexer, aber nützlich bei Hängern)
    # Du kannst multiprocessing-Jobs in Subprozesse mit Timeout verpacken
        # TUNE

        # TEST 

        # SAVE FINISH 

    pass