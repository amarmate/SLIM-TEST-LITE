from skopt.space import Integer, Real

# --------------------------- # 
#    General Configuration    #
# --------------------------- #
N_SPLITS = 4                
N_CV = 4                  
N_SEARCHES_HYPER = 15     
N_RANDOM_STARTS = 10      
NOISE_SKOPT = 1e-3
N_TESTS = 15              
P_TEST = 0.2 
SEED = 0
N_TIME_BINS = 300
SUFFIX_SAVE = '1'
PREFIX_SAVE = 'GP'  
EXPERIMENT_NAME = 'GP_Experiment'

# --------------------------- # 
# GP Experiment Configuration # 
# --------------------------- # 

SELECTORS = ['dalex']
FUNCTIONS = ['add', 'multiply', 'subtract', 'AQ']
STOP_THRESHOLD = 1000
PI = [(2000, 100), (1000, 200), (500, 400)]   # n_generations, pop_size


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