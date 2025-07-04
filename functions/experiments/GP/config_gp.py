from skopt.space import Integer, Real

from slim_gsgp_lib_np.datasets.data_loader import ( 
    load_airfoil, load_boston, load_concrete_strength, load_diabetes, load_efficiency_heating, load_forest_fires,
    load_istanbul, load_ld50, load_bioav, load_parkinson_updrs, load_ppb, load_resid_build_sale_price,
)

from slim_gsgp_lib_np.datasets.synthetic_datasets import (
    load_synthetic1, load_synthetic2, load_synthetic3, load_synthetic4, load_synthetic5, load_synthetic6, 
    load_synthetic7, load_synthetic8, load_synthetic9, load_synthetic10, load_synthetic11, #  load_synthetic12,
)
# --------------------------- # 
#    General Configuration    #
# --------------------------- #

datasets = {name.split('load_')[1] : loader for name, loader in globals().items() if name.startswith('load_') and callable(loader)}

N_SPLITS = 4                
N_CV = 4                  
N_SEARCHES_HYPER = 20     
N_RANDOM_STARTS = 10      
NOISE_SKOPT = 1e-3
N_TESTS = 20              
P_TEST = 0.25 
SEED = 0
N_TIME_BINS = 300
SUFFIX_SAVE = '1'
PREFIX_SAVE = 'GP_STDERR'  
EXPERIMENT_NAME = 'GP_STDERR'
TEST_DIR = 'test'
TUNE_DIR = 'train'

DATA_DIR = 'data'
REPO_URL = 'git@github.com:amarmate/data_transfer.git'
AUTO_COMMIT_INTERVAL = 0.1 * 3600 # every 6 min 


# --------------------------- # 
# GP Experiment Configuration # 
# --------------------------- # 

SELECTORS = ['dalex', 'dalex_fast_rand']
FUNCTIONS = ['add', 'multiply', 'subtract', 'AQ']
# CONSTANTS = constants = [round(i*0.05, 2) for i in range(2, 21)] + [round(-i*0.05, 2) for i in range(2, 21)]  # From -1 to 1 with step 0.05 
CONSTANTS = [round(i*0.1, 2) for i in range(2, 21)] + [round(-i*0.1, 2) for i in range(2, 21)]  # From -2.0 to 2.0 with step 0.1 

STOP_THRESHOLD = 0.2
PI = [(2000, 100), (1000, 200), (500, 400)]   # n_generations, pop_size
PROB_TERMINAL = 0.7
PROB_CONST = 0.2
INIT_DEPTH = 2
STD_ERRS = True


SPACE_PARAMETERS = [
    Integer(6, 9, name='max_depth'),                    
    Integer(0, 2, name='pop_iter_setting', prior='uniform'),                                                                    
    Real(4, 60, name='particularity_pressure', prior='log-uniform'),
    Real(0.5, 0.9, name='p_xo'),    

    # ---------------- not currently used ---------------- #
    # Integer(2, 4, name='init_depth'),
    # Real(0.10, 0.25, name='prob_const'), 
    # Real(0.5, 0.9, name='prob_terminal'),             
]


# --------------------------- #
#    Save Configuration       #
# --------------------------- #

gen_params = { 
    "test_elite": False,
    "dataset_name": "test",
    "init_depth": INIT_DEPTH,
    "prob_const": PROB_CONST,
    "prob_terminal": PROB_TERMINAL,
    "tree_functions": FUNCTIONS,
    "tree_constants": CONSTANTS,
    "std_errs": STD_ERRS,
    "log_level": 'evaluate',
    "it_tolerance": STOP_THRESHOLD,
    "down_sampling": 1,
    "full_return": True,
    "verbose": False,
}


config = {
    'N_SPLITS': N_SPLITS,
    'N_CV': N_CV,
    'N_SEARCHES_HYPER': N_SEARCHES_HYPER,
    'N_RANDOM_STARTS': N_RANDOM_STARTS,
    'NOISE_SKOPT': NOISE_SKOPT,
    'N_TESTS': N_TESTS,
    'P_TEST': P_TEST,
    'SEED': SEED,
    'N_TIME_BINS': N_TIME_BINS,
    'SUFFIX_SAVE': SUFFIX_SAVE,
    'PREFIX_SAVE': PREFIX_SAVE,
    'EXPERIMENT_NAME': EXPERIMENT_NAME,
    'AUTO_COMMIT_INTERVAL': AUTO_COMMIT_INTERVAL,

    'DATA_DIR': DATA_DIR,
    'TEST_DIR': TEST_DIR,
    'TUNE_DIR': TUNE_DIR,
    'REPO_URL': REPO_URL,

    'SELECTORS': SELECTORS,
    'FUNCTIONS': FUNCTIONS,
    'STOP_THRESHOLD': STOP_THRESHOLD,
    'PI': [PI],
    'datasets' : datasets,

    'SPACE_PARAMETERS': SPACE_PARAMETERS,
    'gen_params' : gen_params,
    'multi_run'  : False
}