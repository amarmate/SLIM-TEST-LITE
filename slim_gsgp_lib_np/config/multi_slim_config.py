import numpy as np 
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any

# SLIM GSGP
from slim_gsgp_lib_np.initializers.initializers import rhh, grow, full, simple
from slim_gsgp_lib_np.algorithms.GSGP.operators.crossover_operators import geometric_crossover
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.operators.crossovers import *
from slim_gsgp_lib_np.selection.selection_algorithms import tournament_selection_min
from slim_gsgp_lib_np.evaluators.fitness_functions import *
from slim_gsgp_lib_np.utils.utils import (get_best_min, protected_div)

# MULTI SLIM
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree_utils import initializer
from slim_gsgp_lib_np.evaluators.fitness_functions import *

# --------------------------- FUNCTIONS AND CONSTANTS ---------------------------
FUNCTIONS = ['add','subtract','multiply','divide']
CONSTANTS = [round(-1 + (2 * i) / (100 - 1), 2) for i in range(100) if np.abs(i) > 0.1]

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
       - p_xo: Crossover probability (default 0.0).
       - p_inflate: Inflate mutation probability (default 0.2).
       - p_struct: Structural mutation probability (default 0.0).
       - mode: Distribution to choose the depth of the new tree for sturcture mutation 
               (default: "exp"), options: "normal", "exp", "uniform".
       - pop_size: Population size (default 100).
       - p_struct_xo: Structural crossover probability (default 0.0).
       - decay_rate: Decay rate for mutation probabilities (default 0.1).
       - mut_xo_operator: Mutation operator for crossover (default "rshuffle").
       - n_cases: Number of cases used in lexicase selection (default 10).
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
    n_cases: int = 10
    p_xo: float = 0.0
    p_inflate: float = 0.2
    p_struct: float = 0.0
    mode: float = 'exp'
    pop_size: int = 100
    p_struct_xo: float = 0.0
    decay_rate: float = 0.1
    mut_xo_operator: str = "rshuffle"
    callbacks: Optional[list] = None
    
    # SLIM_GSGP_PI_INIT parameters
    tree_functions: list = field(default_factory=lambda: FUNCTIONS)
    tree_constants: list = field(default_factory=lambda: CONSTANTS)
    prob_const: float = 0.2
    prob_terminal: float = 0.7
    init_depth: int = 6
    
    # SLIM_GSGP_SOLVE_PARAMETERS
    max_depth: int = 15
    reconstruct: bool = True
    n_iter: int = 1000
    elitism: bool = True
    n_elites: int = 1

@dataclass
class GPParameters:
    """
    Parameters for running the GP algorithm.
    
    This data class encapsulates the parameters required to configure the Genetic Programming (GP) algorithm.
    
    Attributes:
        pop_size (int): Population size.
        n_iter (int): Number of iterations in the solve stage.
        p_xo (float): Crossover probability.
        elitism (bool): Flag indicating whether elitism is applied.
        n_elites (int): Number of elite individuals to retain.
        selector (str): Selection method used in the algorithm.
        n_cases (int): Number of cases used in lexicase selection.
        epsilon (float): epsilon value to use for manual epsilon lexicase selection. 
        max_depth (Optional[int]): Maximum allowed tree depth.
        init_depth (int): Initial depth used for generating individuals.
        initializer (str): Initialization method.
        prob_const (float): Probability of choosing a constant during initialization.
        prob_terminal (float): Terminal probability during initialization.
        tree_functions (List): List of available function nodes (derived from the keys of FUNCTIONS).
        tree_constants (List): List of available constant nodes (derived from CONSTANTS, with some parsing).
        callbacks (List): List of callbacks to be executed during the algorithm.
        particularity_pressure (float): Pressure to apply to the particularity of the individuals (dalex).
    """

    pop_size: int = 100
    n_iter: int = 1000
    p_xo: float = 0.2
    elitism: bool = True
    n_elites: int = 1
    selector: str = "e_lexicase"
    n_cases: int = 10
    epsilon: float = 1e-6,
    max_depth: Optional[int] = 15
    init_depth: int = 6
    initializer: str = "rhh"
    prob_const: float = 0.2
    prob_terminal: float = 0.7
    tree_functions: list = field(default_factory=lambda: FUNCTIONS)
    tree_constants: list = field(default_factory=lambda: CONSTANTS)
    callbacks: list = None
    particularity_pressure: float = 20

# ---------------------------- MULTI SLIM SOLVE parameters ----------------------------
multi_solve_params = {
    "run_info": None,
    "ffunction": "rmse",
    "n_iter": 1000,
    "elitism": True,
    "n_elites": 1,
    "log": 0,
    "verbose": 1,
    "test_elite": False,
    "timeout": None
}

# ---------------------------- MULTI SLIM parameters ----------------------------
multi_params = {
    "selector": 'tournament',
    "mutator": None,
    "xo_operator": None,
    "p_mut": 0.2,
    "find_elit_func": get_best_min,
    "seed": 74,
    "decay_rate": 0.1,
}

# ---------------------------- MULTI SLIM PI_INIT parameters ----------------------------
multi_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "depth_condition": 3,
    "max_depth": 3,
    "p_c": 0.2,
    "p_t": 0.7, 
    "p_specialist": 0.7,
    "pop_size" : 100,
}

fitness_function_options = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "mae_int": mae_int,
    "signed_errors": signed_errors
}
