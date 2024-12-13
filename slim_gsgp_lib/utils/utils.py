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
import math
import random

import numpy as np
import torch
from slim_gsgp_lib.algorithms.GP.representations.tree_utils import (create_full_random_tree,
                                                                create_grow_random_tree)
from slim_gsgp_lib.algorithms.GSGP.representations.tree import Tree
from sklearn.metrics import root_mean_squared_error


def protected_div(x1, x2):
    """Implements the division protected against zero denominator

    Performs division between x1 and x2. If x2 is (or has) zero(s), the
    function returns the numerator's value(s).

    Parameters
    ----------
    x1 : torch.Tensor
        The numerator.
    x2 : torch.Tensor
        The denominator.

    Returns
    -------
    torch.Tensor
        Result of protected division between x1 and x2.
    """
    return torch.where(
        torch.abs(x2) > 0.001,
        torch.div(x1, x2),
        torch.tensor(1.0, dtype=x2.dtype, device=x2.device),
    )


def mean_(x1, x2):
    """
    Compute the mean of two tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        The first tensor.
    x2 : torch.Tensor
        The second tensor.

    Returns
    -------
    torch.Tensor
        The mean of the two tensors.
    """
    return torch.div(torch.add(x1, x2), 2)


def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """Splits X and y tensors into train and test subsets

    This method replicates the behaviour of Sklearn's 'train_test_split'.

    Parameters
    ----------
    X : torch.Tensor
        Input data instances,
    y : torch.Tensor
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random numbers generators.

    Returns
    -------
    X_train : torch.Tensor
        Training data instances.
    y_train : torch.Tensor
        Training target vector.
    X_test : torch.Tensor
        Test data instances.
    y_test : torch.Tensor
        Test target vector.
    train_indices : torch.Tensor
        Indices representing the training partition.
    test_indices : torch.Tensor
        Indices representing the test partition.
    """
    torch.manual_seed(seed)
    if shuffle:
        indices = torch.randperm(X.shape[0])
    else:
        indices = torch.arange(0, X.shape[0], 1)
    split = int(math.floor(p_test * X.shape[0]))
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test


def tensor_dimensioned_sum(dim):
    """
    Generate a sum function over a specified dimension.

    Parameters
    ----------
    dim : int
        The dimension to sum over.

    Returns
    -------
    function
    A function that sums tensors over the specified dimension.
    """

    def tensor_sum(input):
        return torch.sum(input, dim)

    return tensor_sum

def verbose_reporter(params, 
                     first=False, 
                     precision=3,
                     col_width=20):
    """
    Prints a formatted report of custom parameters.

    Parameters
    ----------
    params : dict
        A dictionary containing key-value pairs of parameters to be reported.
    first : bool, default=False
        Whether this is the first report to be printed.
    precision : int, default=3
        The number of decimal places to display for float values.
    col_width : int, default=20
        The width of the columns in the report.

    Returns
    -------
    None
        Outputs a formatted report to the console.
    """
    if first: 
        separator = ("+" + "-" * (col_width + 3)) * len(params) + "+"
        print(separator)
        print("".join([f"|{key.center(col_width+3)}" for key in params.keys()]) + "|")
        print(separator)

    # Print values
    values = []
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        elif value is None:
            formatted_value = "None"
        else:
            formatted_value = str(value)
        values.append(formatted_value.center(col_width+3))
    
    print("|" + "".join([f"{value}|" for value in values]))
    print(('|' + '-' * (col_width + 3))*len(params) + '|')



def get_terminals(X):
    """
    Get terminal nodes for a dataset.

    Parameters
    ----------
    X : (torch.Tensor)
        An array to get the set of TERMINALS from, it will correspond to the columns.

    Returns
    -------
    dict
        Dictionary of terminal nodes.
    """

    return  {f"x{i}": i for i in range(len(X[0]))}


def get_best_min(population, n_elites):
    """
    Get the best individuals from the population with the minimum fitness.

    Parameters
    ----------
    population : Population
        The population of individuals.
    n_elites : int
        Number of elites to return.

    Returns
    -------
    list
        The list of elite individuals.
    Individual
        Best individual from the elites.
    """
    if n_elites > 1:
        idx = np.argpartition(population.fit, n_elites)
        elites = [population.population[i] for i in idx[:n_elites]]
        return elites, elites[np.argmin([elite.fitness for elite in elites])]

    else:
        elite = population.population[np.argmin(population.fit)]
        return [elite], elite


def get_best_max(population, n_elites):
    """
    Get the best individuals from the population with the maximum fitness.

    Parameters
    ----------
    population : Population
        The population of individuals.
    n_elites : int
        Number of elites to return.

    Returns
    -------
    list
        The list of elite individuals.
    Individual
        Best individual from the elites.
    """
    if n_elites > 1:
        idx = np.argpartition(population.fit, -n_elites)
        elites = [population.population[i] for i in idx[-n_elites:]]
        return elites, elites[np.argmax([elite.fitness for elite in elites])]

    else:
        elite = population.population[np.argmax(population.fit)]
        return [elite], elite

def get_random_tree(
        max_depth,
        FUNCTIONS,
        TERMINALS,
        CONSTANTS,
        inputs=None,
        p_c=0.3,
        grow_probability=1,
        logistic=True,
        list_form=False,
):
    """
    Get a random tree using either grow or full method.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    FUNCTIONS : dict
        Dictionary of functions.
    TERMINALS : dict
        Dictionary of terminals.
    CONSTANTS : dict
        Dictionary of constants.
    inputs : torch.Tensor
        Input tensor for calculating semantics.
    p_c : float, default=0.3
        Probability of choosing a constant.
    grow_probability : float, default=1
        Probability of using the grow method.
    logistic : bool, default=True
            Whether to use logistic semantics.
    list_form : bool, default=False
        Whether the functions, terminals, and constants are in list form.

    Returns
    -------
    Tree
        The generated random tree.
    """
    if random.random() < grow_probability:
        tree_structure = create_grow_random_tree(
            max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=p_c,
        )
    else:
        tree_structure = create_full_random_tree(
            max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=p_c
        )

    tree = Tree(
        structure=tree_structure,
        train_semantics=None,
        test_semantics=None,
        reconstruct=True,
    )
    if inputs is not None:
        tree.calculate_semantics(inputs, testing=False, logistic=logistic)
    return tree


def generate_random_uniform(lower, upper):
    """
    Generate a random number within a specified range using numpy random.uniform.

    Parameters
    ----------
    lower : float
        The lower bound of the range for generating the random number.
    upper : float
        The upper bound of the range for generating the random number.

    Returns
    -------
    Callable
        A function that when called, generates a random number within the specified range.
    Notes
    -----
    The returned function takes no input and returns a random float between lower and upper whenever called.
    """

    def generate_num():
        """
        Generate a random number within a specified range.

        Returns
        -------
        float
            A random number between the defined lower and upper bounds.
        """
        return random.uniform(lower, upper)

    generate_num.lower = lower
    generate_num.upper = upper
    return generate_num


def show_individual(tree, operator):
    """
    Display an individual's structure with a specified operator.

    Parameters
    ----------
    tree : Tree
        The tree representing the individual.
    operator : str
        The operator to display ('sum' or 'prod').

    Returns
    -------
    str
        The string representation of the individual's structure.
    """
    op = "+" if operator == "sum" else "*"

    return f" {op} ".join(
        [
            (
                str(t.structure)
                if isinstance(t.structure, tuple)
                else (
                    f"f({t.structure[1].structure})"
                    if len(t.structure) == 3
                    else f"f({t.structure[1].structure} - {t.structure[2].structure})"
                )
            )
            for t in tree.collection
        ]
    )


def gs_rmse(y_true, y_pred):
    """
    Calculate the root mean squared error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        The root mean squared error.
    """
    return root_mean_squared_error(y_true, y_pred[0])


def gs_size(y_true, y_pred):
    """
    Get the size of the predicted values.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    int
        The size of the predicted values.
    """
    return y_pred[1]


def validate_inputs(X_train, y_train, X_test, y_test, pop_size, n_iter, elitism, n_elites, init_depth, log_path,
                    prob_const, tree_functions, tree_constants, log, verbose, minimization, n_jobs, test_elite,
                    fitness_function, initializer, tournament_size, ms_lower, ms_upper, p_inflate, p_struct,
                    depth_distribution,):
    """
    Validates the inputs based on the specified conditions.

    Parameters
    ----------
    tournament_size
    X_train: (torch.Tensor)
        Training input data.
    y_train: (torch.Tensor)
        Training output data.
    X_test: (torch.Tensor), optional
        Testing input data.
    y_test: (torch.Tensor), optional
        Testing output data.
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    elitism : bool, optional
        Indicate the presence or absence of elitism.
    n_elites : int, optional
        The number of elites.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    log_path : str, optional
        The path where is created the log directory where results are saved.
    log : int, optional
        Level of detail to utilize in logging.
    verbose : int, optional
       Level of detail to include in console output.
    minimization : bool, optional
        If True, the objective is to minimize the fitness function. If False, maximize it (default is True).
    fitness_function : str, optional
        The fitness function used for evaluating individuals (default is from gp_solve_parameters).
    initializer : str, optional
        The strategy for initializing the population (e.g., "grow", "full", "rhh").
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).
    prob_const : float, optional
        The probability of introducing constants into the trees during evolution.
    tree_functions : list, optional
        List of allowed functions that can appear in the trees Check documentation for the available functions.
    tree_constants : list, optional
        List of constants allowed to appear in the trees.
    test_elite : bool, optional
        Whether to test the elite individual on the test set after each generation.
    ms_lower : int, optional
        The lower bound for the maximum size of the trees.
    ms_upper : int, optional
        The upper bound for the maximum size of the trees.
    p_inflate : float, optional
        The probability of inflating a tree.
    p_struct : float, optional
        The probability of structural mutation.
    depth_distribution : list, optional
        Distribution to choose the depth of the new tree (default: "norm"), options: "norm", "exp", "uniform", "max".


    """
    if not isinstance(X_train, torch.Tensor):
        raise TypeError("X_train must be a torch.Tensor")
    if not isinstance(y_train, torch.Tensor):
        raise TypeError("y_train must be a torch.Tensor")
    if X_test is not None and not isinstance(X_test, torch.Tensor):
        raise TypeError("X_test must be a torch.Tensor")
    if y_test is not None and not isinstance(y_test, torch.Tensor):
        raise TypeError("y_test must be a torch.Tensor")
    if not isinstance(pop_size, int):
        raise TypeError("pop_size must be an int")
    if not isinstance(n_iter, int):
        raise TypeError("n_iter must be an int")
    if not isinstance(elitism, bool):
        raise TypeError("elitism must be a bool")
    if not isinstance(n_elites, int):
        raise TypeError("n_elites must be an int")
    if not isinstance(init_depth, int):
        raise TypeError("init_depth must be an int")
    if not isinstance(log_path, str):
        raise TypeError("log_path must be a str")
    if not isinstance(tournament_size, int):
        raise TypeError("tournament_size must be an int")

    # assuring the prob_const is valid
    if not (isinstance(prob_const, float) or isinstance(prob_const, int)):
        raise TypeError("prob_const must be a float (or an int when probability is 1 or 0)")

    if not 0 <= prob_const <= 1:
        raise ValueError("prob_const must be a number between 0 and 1")

    if n_iter < 1:
        raise ValueError("n_iter must be greater than 0")

    # Ensuring the functions and constants passed are valid
    if not isinstance(tree_functions, list) or len(tree_functions) == 0:
        raise TypeError("tree_functions must be a non-empty list")

    if not isinstance(tree_constants, list) or len(tree_constants) == 0:
        raise TypeError("tree_constants must be a non-empty list")

    assert all(isinstance(elem, (int, float)) and not isinstance(elem, bool) for elem in tree_constants), \
    "tree_constants must be a list containing only integers and floats"

    if not isinstance(log, int):
        raise TypeError("log_level must be an int")

    assert 0 <= log <= 4, "log_level must be between 0 and 4"

    if not isinstance(verbose, int):
        raise TypeError("verbose level must be an int")

    assert 0 <= verbose <= 1, "verbose level must be either 0 or 1"

    if not isinstance(minimization, bool):
        raise TypeError("minimization must be a bool")

    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an int")

    assert n_jobs >= 1, "n_jobs must be at least 1"

    if not isinstance(test_elite, bool):
        raise TypeError("test_elite must be a bool")

    if not isinstance(fitness_function, str):
        raise TypeError("fitness_function must be a str")

    if not isinstance(initializer, str):
        raise TypeError("initializer must be a str")

    if tournament_size < 2:
        raise ValueError("tournament_size must be at least 2")

    if ms_lower == ms_upper:
        raise ValueError("ms_lower and ms_upper must be different")
    
    if ms_lower > ms_upper:
        raise ValueError("ms_lower must be smaller than ms_upper")
    
    if not isinstance(p_inflate, float) and p_inflate != 0:
        raise TypeError("p_inflate must be a float")
    
    if not isinstance(p_struct, float) and p_struct != 0:
        raise TypeError("p_struct must be a float")
    
    if p_inflate < 0 or p_struct < 0:
        raise ValueError("p_inflate and p_struct must be greater or equal to 0")
    
    if p_inflate + p_struct > 1:
        raise ValueError("p_inflate + p_struct must be smaller or equal to 1")
    
    if depth_distribution not in ["norm", "exp", "uniform", "max"]:
        raise ValueError("depth_distribution must be one of: 'norm', 'exp', 'uniform', 'max'")


def check_slim_version(slim_version):
    """
    Validate the slim_gsgp version given as input bu the users and assign the correct values to the parameters op, sig and trees
    Parameters
    ----------
    slim_version : str
        Name of the slim_gsgp version.

    Returns
    -------
    op, sig, trees
        Parameters reflecting the kind of operation considered, the use of the sigmoid and the use of multiple trees.
    """
    if slim_version == "SLIM+SIG2":
        return "sum", True, True
    elif slim_version == "SLIM*SIG2":
        return "mul", True, True
    elif slim_version == "SLIM+ABS":
        return "sum", False, False
    elif slim_version == "SLIM*ABS":
        return "mul", False, False
    elif slim_version == "SLIM+SIG1":
        return "sum", True, False
    elif slim_version == "SLIM*SIG1":
        return "mul", True, False
    else:
        raise Exception('Invalid SLIM configuration')

def _evaluate_slim_individual(individual, ffunction, y, testing=False, operator="sum"):
    """
    Evaluate the individual using a fitness function.

    Args:
        ffunction: Fitness function to evaluate the individual.
        y: Expected output (target) values as a torch tensor.
        testing: Boolean indicating if the evaluation is for testing semantics.
        operator: Operator to apply to the semantics ("sum" or "prod").

    Returns:
        None
    """
    if operator == "sum":
        operator = torch.sum
    else:
        operator = torch.prod

    if testing:
        individual.test_fitness = ffunction(
            y,
            torch.clamp(
                operator(individual.test_semantics, dim=0),
                -1000000000000.0,
                1000000000000.0,
            ),
        )

    else:
        individual.fitness = ffunction(
            y,
            torch.clamp(
                operator(individual.train_semantics, dim=0),
                -1000000000000.0,
                1000000000000.0,
            ),
        )

        # if testing is false, return the value so that training parallelization has effect
        return ffunction(
                y,
                torch.clamp(
                    operator(individual.train_semantics, dim=0),
                    -1000000000000.0,
                    1000000000000.0,
                ),
            )
    

def get_indices(tree, path=()):
    """
    Get all indices that can be used to access valid subtrees or terminal nodes in a tree.

    Parameters
    ----------
    tree : tuple
        The current node of the tree.
    path : tuple
        The path to the current node.

    Returns
    -------
    list
        A list of all indices that can be used to access valid elements in the tree.
    """
    indices = []
    
    # If not tuple 
    if not isinstance(tree, tuple):
        indices.append(path)

    # If tuple, separate
    else:
        indices.append(path) if path != () else None
        op, left, right = tree  
        # Can substitute the left or right
        indices.extend(get_indices(left, path + (1,)))
        indices.extend(get_indices(right, path + (2,)))

    return indices

def get_indices_with_levels(tree):
    """
    Get all indices that can be used to access valid subtrees or terminal nodes in a tree,
    along with their corresponding levels in the tree.

    Parameters
    ----------
    tree : tuple
        The root node of the tree.

    Returns
    -------
    list
        A list of tuples, each containing an index path and its corresponding level in the tree.
    """
    
    def traverse(sub_tree, path=(), level=0):
        indices_with_levels = []
        
        # If not tuple 
        if not isinstance(sub_tree, tuple):
            indices_with_levels.append((path, level))

        # If tuple, separate
        else:
            indices_with_levels.append((path, level)) if path != () else None
            op, left, right = sub_tree  
            # Can substitute the left or right
            indices_with_levels.extend(traverse(left, path + (1,), level + 1))
            indices_with_levels.extend(traverse(right, path + (2,), level + 1))

        return indices_with_levels
    
    return traverse(tree)


def swap_sub_tree(tree, new_tree, indices):
    """
    Swap a subtree in a tree.

    Parameters
    ----------
    tree : tuple
        The current node of the tree.
    new_tree : tuple        
        The new subtree to be swapped.
    indices : list
        The indices of the tree to be swapped.
    """
    if indices == []:
        return new_tree
    
    index = indices[0]
    return tree[:index] + (swap_sub_tree(tree[index],new_tree, indices[1:]),) + tree[index+1:]


def get_subtree(tree, indices):
        """
        Get a subtree in a tree.

        Parameters
        ----------
        tree : tuple
            The current node of the tree.
        indices : list
            The indices of the tree to be accessed.

        Returns
        -------
        tuple
            The subtree in the tree.
        """
        if indices == []:
            return tree

        index = indices[0]
        return get_subtree(tree[index], indices[1:])