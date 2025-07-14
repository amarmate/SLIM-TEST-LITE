from skopt.space import Integer, Real
from slim_gsgp_lib_np.datasets.synthetic_datasets import (
    load_synthetic1, load_synthetic2, load_synthetic3, load_synthetic4, load_synthetic5, load_synthetic6, 
    load_synthetic7, load_synthetic8, load_synthetic9, load_synthetic10, load_synthetic11, # load_synthetic12,
)
# from slim_gsgp_lib_np.datasets.data_loader import ( 
#     load_airfoil, load_boston, load_concrete_strength, load_diabetes, load_efficiency_heating, load_forest_fires,
#     load_istanbul, load_ld50, load_bioav, load_parkinson_updrs, load_ppb, load_resid_build_sale_price,
# )

# --------------------------- # 
#    General Configuration    #
# --------------------------- #

SEED = 0

datasets = {name.split('load_')[1] : loader for name, loader in globals().items() if name.startswith('load_') and callable(loader)}

N_SPLITS = 4                
N_CV = 4                     # 4      

N_SEARCHES_HYPER_GP = 20      # 20 
N_RANDOM_STARTS_GP = 10       # 10

N_SEARCHES_HYPER_MULTI = 20   # 20
N_RANDOM_STARTS_MULTI = 10    # 10

NOISE_SKOPT = 1e-3
N_TESTS = 20              
P_TEST = 0.25 
N_TIME_BINS = 300
SUFFIX_SAVE = '3'
PREFIX_SAVE = 'MULTI'  
EXPERIMENT_NAME = 'MULTI'
TEST_DIR = 'test'
TUNE_DIR = 'train'

DATA_DIR = 'data'
REPO_URL = 'git@github.com:amarmate/data_transfer.git'
AUTO_COMMIT_INTERVAL = 0.25 * 3600 # every 15 min  


# ------------------------------ # 
#  GP Experiment Configuration   # 
# ------------------------------ # 

SELECTOR_GP = 'dalex_fast_rand'
FUNCTIONS_GP = ['add', 'multiply', 'subtract', 'AQ']
CONSTANTS_GP = [round(i*0.1, 2) for i in range(2, 21)] + [round(-i*0.1, 2) for i in range(2, 21)]
STOP_THRESHOLD_GP = 0.3 # 0.2 
# PI_GP = [(10, 10), (10, 10), (10, 10),]   # n_generations, pop_size
PI_GP = [
    # (2000, 100),
    (1000, 200), (400, 400),(150, 1000)
    ]   # n_generations, pop_size

PROB_TERMINAL_GP = 0.7
PROB_CONST_GP = 0.2
INIT_DEPTH_GP = 2

SPACE_PARAMETERS_GP = [
    Integer(4, 9, name='max_depth_gp'),                    
    Integer(0, 2, name='pop_iter_setting_gp', prior='uniform'),                                                                    
    Real(2, 40, name='particularity_pressure_gp', prior='uniform'),
    Real(0.4, 0.9, name='p_xo_gp'),    
]

gp_params = { 
    "init_depth_gp": INIT_DEPTH_GP,
    "prob_const_gp": PROB_CONST_GP,
    "prob_terminal_gp": PROB_TERMINAL_GP,
    "tree_functions_gp": FUNCTIONS_GP,
    "tree_constants_gp": CONSTANTS_GP,
    "log_level_gp": 0,
    "it_tolerance_gp": STOP_THRESHOLD_GP,
    "down_sampling_gp": 1,
    "selector_gp" : SELECTOR_GP,
}

# ------------------------------ # 
# MULTI Experiment Configuration # 
# ------------------------------ # 

SELECTOR_MULTI = 'dalex'
FUNCTIONS_MULTI = ['add', 'multiply', 'subtract', 'AQ']
CONSTANTS_MULTI = [round(i*0.1, 2) for i in range(2, 21)] + [round(-i*0.1, 2) for i in range(2, 21)]

STOP_THRESHOLD_MULTI = 0.3  # CHANGE 
PI_MULTI = [(2000, 100), (1000, 200), (500, 400)]   # n_generations, pop_size
# PI_MULTI = [(20, 10), (10, 20), (5, 4)]   # n_generations, pop_size
PROB_TERMINAL_MULTI = 0.7
PROB_CONST_MULTI = 0.2

SPACE_PARAMETERS_MULTI = [
    Integer(2, 5, name='max_depth'),     
    Integer(2, 7, name='depth_condition'),                
    Integer(0, 2, name='pop_iter_setting', prior='uniform'),                                                                    
    Real(1, 30, name='particularity_pressure', prior='uniform'),
    Real(0.3, 0.75, name='p_xo'),    
]

MULTI_MAX_DEPTH = 3
MULTI_DEPTH_CONDITION = 6
MULTI_PP = 2.5
MULTI_XO = 0.5
MULTI_POP_SIZE = 100  # 100
MULTI_N_ITER = 2000    # 2000

multi_params = {
    "test_elite"            : False,
    "dataset_name"          : "test",
    "prob_const"            : PROB_CONST_MULTI,
    "prob_terminal"         : PROB_TERMINAL_MULTI,
    "ensemble_functions"    : FUNCTIONS_MULTI,
    "ensemble_constants"    : CONSTANTS_MULTI,
    "log_level"             : 0,
    "it_tolerance"          : STOP_THRESHOLD_MULTI,
    "down_sampling"         : 1,
    "full_return"           : True,
    "verbose"               : True,  # FALSE 
    "gp_version"            : 'gp',
    "selector"              : SELECTOR_MULTI,

    "max_depth"             : MULTI_MAX_DEPTH,
    "depth_condition"       : MULTI_DEPTH_CONDITION,
    "particularity_pressure": MULTI_PP,
    "p_xo"                  : MULTI_XO, 
    "pop_size"              : MULTI_POP_SIZE,
    "n_iter"                : MULTI_N_ITER,
}

multi_params.update(gp_params)

# --------------------------- #
#    Save Configuration       #
# --------------------------- #



config = {
    'N_SPLITS': N_SPLITS,
    'N_CV': N_CV,

    'N_SEARCHES_HYPER_GP': N_SEARCHES_HYPER_GP,
    'N_RANDOM_STARTS_GP': N_RANDOM_STARTS_GP,
    'N_SEARCHES_HYPER_MULTI': N_SEARCHES_HYPER_MULTI,
    'N_RANDOM_STARTS_MULTI': N_RANDOM_STARTS_MULTI,

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

    'SELECTOR_GP': SELECTOR_GP,
    'SELECTOR_MULTI': SELECTOR_MULTI,
    'SELECTORS' : [SELECTOR_MULTI], 

    'FUNCTIONS_GP': FUNCTIONS_GP,
    'FUNCTIONS_MULTI': FUNCTIONS_MULTI,
        
    'PI' : [PI_MULTI, PI_GP],
    'datasets' : datasets,

    'SPACE_GP': SPACE_PARAMETERS_GP,
    'SPACE_MULTI': SPACE_PARAMETERS_MULTI,

    'gen_params' : multi_params,
    'multi_run' : True,
}