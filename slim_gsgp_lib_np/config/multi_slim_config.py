import numpy as np 
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any

# SLIM GSGP
from slim_gsgp_lib_np.initializers.initializers import rhh, grow, full, simple
from slim_gsgp_lib_np.algorithms.GSGP.operators.crossover_operators import geometric_crossover
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.operators.crossovers import xo_operator
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.operators.mutators import (deflate_mutation, inflate_mutation)
from slim_gsgp_lib_np.selection.selection_algorithms import tournament_selection_min
from slim_gsgp_lib_np.evaluators.fitness_functions import *
from slim_gsgp_lib_np.utils.utils import (get_best_min, protected_div)

# MULTI SLIM
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree_utils import initializer
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.operators.xo import crossover
from slim_gsgp_lib_np.evaluators.fitness_functions import *

# --------------------------- FUNCTIONS AND CONSTANTS ---------------------------
FUNCTIONS = {
    'add': {'function': np.add, 'arity': 2},
    'subtract': {'function': np.subtract, 'arity': 2},
    'multiply': {'function': np.multiply, 'arity': 2},
    'divide': {'function': protected_div, 'arity': 2}
}

constants = [round(-1 + (2 * i) / (100 - 1), 2) for i in range(100) if np.abs(i) > 0.1]
CONSTANTS = {f'c_{i}': lambda _: np.array(i) for i in constants}


# ---------------------------- SLIM GSGP parameters ----------------------------
@dataclass
class SlimParameters:
    """
    Parameters for running the SLIM_GSGP algorithm.
    
    The SLIM_GSGP algorithm is configured with three groups of parameters:
    
    1. SLIM_GSGP_PARAMETERS: Parameters for the main algorithm (mutation, crossover, selection, etc.).
       - slim_version: Version of the SLIM algorithm (default "SLIM*SIG1").
       - initializer: Initialization method (default "rhh").
       - selector: Selection method (default "e_lexicase").
       - deflate_mutator: Function for deflate mutation (default deflate_mutation).
       - xo_operator: Crossover operator (default xo_operator).
       - p_xo: Crossover probability (default 0.0).
       - find_elit_func: Function to find the elite individual (default get_best_min).
       - p_inflate: Inflate mutation probability (default 0.2).
       - p_struct: Structural mutation probability (default 0.0).
       - pop_size: Population size (default 100).
       - p_struct_xo: Structural crossover probability (default 0.0).
       - decay_rate: Decay rate for mutation probabilities (default 0.1).
       - mut_xo_operator: Mutation operator for crossover (default "rshuffle").
       - callbacks: List of callbacks (default None).   
    
    2. SLIM_GSGP_PI_INIT: Parameters for initializing the candidate solutions.
       - FUNCTIONS: Dictionary of function nodes.
       - CONSTANTS: Dictionary of constant nodes.
       - p_c: Probability of choosing a constant (default 0.2).
       - p_t: Terminal probability (default 0.7).
       - init_depth: Initial tree depth (default 6).
    
    3. SLIM_GSGP_SOLVE_PARAMETERS: Parameters for the solve stage.
       - run_info: Run information (default None).
       - max_depth: Maximum allowed tree depth (default 15).
       - reconstruct: Whether to reconstruct individuals (default True).
       - n_iter: Number of iterations (default 1000).
       - elitism: Whether elitism is used (default True).
       - n_elites: Number of elites (default 1).
    """
    # SLIM_GSGP_PARAMETERS
    initializer: str = "rhh"
    selector: str = "e_lexicase"
    deflate_mutator: Callable = deflate_mutation
    xo_operator: Callable = xo_operator
    p_xo: float = 0.0
    find_elit_func: Callable = get_best_min
    p_inflate: float = 0.2
    p_struct: float = 0.0
    pop_size: int = 100
    p_struct_xo: float = 0.0
    decay_rate: float = 0.1
    mut_xo_operator: str = "rshuffle"
    callbacks: Optional[list] = None
    
    # SLIM_GSGP_PI_INIT parameters
    FUNCTIONS: Dict[str, Any] = field(default_factory=lambda: FUNCTIONS)
    CONSTANTS: Dict[str, Callable] = field(default_factory=lambda: CONSTANTS)
    p_c: float = 0.2
    p_t: float = 0.7
    init_depth: int = 6
    
    # SLIM_GSGP_SOLVE_PARAMETERS
    max_depth: int = 15
    reconstruct: bool = True
    n_iter: int = 1000
    elitism: bool = True
    n_elites: int = 1





# ---------------------------- MULTI SLIM SOLVE parameters ----------------------------
multi_solve_params = {
    "run_info": None,
    "ffunction": "rmse",
    "max_depth": 15,
    "n_iter": 1000,
    "elitism": True,
    "n_elites": 1,
    "log": 1,
    "verbose": 1,
    "test_elite": True,
    "timeout": None
}

# ---------------------------- MULTI SLIM parameters ----------------------------
multi_params = {
    "selector": 'tournament',
    "mutator": None,
    "xo_operator": crossover,
    "p_mut": 0.2,
    "find_elit_func": get_best_min,
    "pop_size": 100,
    "seed": 74,
    "decay_rate": 0.1,
}

# ---------------------------- MULTI SLIM PI_INIT parameters ----------------------------
multi_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "depth_condition": 3,
    "depth_tree": 3,
    "p_c": 0.2,
    "p_t": 0.7, 
    "p_specialist": 0.7,
}

fitness_function_options = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "mae_int": mae_int,
    "signed_errors": signed_errors
}
