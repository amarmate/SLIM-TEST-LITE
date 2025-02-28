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
Selection operator implementation.
"""

import random
import numpy as np

def selector(problem='min', 
             type='tournament', 
             pool_size=2, 
             eps_fraction=1e-4,
             targets=None):
    """
    Returns a selection function based on the specified problem and selection type.

    Parameters
    ----------
    problem : str, optional
        The type of problem to solve. Can be 'min' or 'max'. Defaults to 'min'.
    type : str, optional
        The type of selection to perform. Can be 'tournament', 'e_lexicase', 'lexicase', 'roulette', 'rank_based' or 'tournament_size'.
        Defaults to 'tournament'.
    pool_size : int, optional
        Number of individuals participating in the tournament. Defaults to 2.
    eps_fraction : float, optional
        The fraction of the populations' standard deviation to use as the epsilon threshold. Defaults to 1e-4.
    targets : torch.Tensor, optional
        The true target values for each entry in the dataset. Required for lexicase selection and epsilon lexicase
        selection. Defaults to None.
    pressure_size : float, optional
        Pressure for size in rank selection. Defaults to 1e-4.

    Returns
    -------
    Callable
        A selection function that selects an individual from a population based on the specified problem and selection
        type.
    """
    if problem == 'min':
        if type == 'tournament':
            return tournament_selection_min(pool_size)
        elif type == 'e_lexicase':
            return epsilon_lexicase_selection(targets, eps_fraction, mode='min')
        elif type == 'lexicase':
            return lexicase_selection(targets, mode='min')
        elif type == 'rank_based':
            return rank_based(mode='min', pool_size=pool_size)
        elif type == 'roulette':
            return roulette_wheel_selection
        elif type == 'tournament_size':
            return tournament_selection_min_size(pool_size, pressure_size=0.5)
        else:
            raise ValueError(f"Invalid selection type: {type}")
    elif problem == 'max':
        if type == 'tournament':
            return tournament_selection_max(pool_size)
        elif type == 'e_lexicase':
            return epsilon_lexicase_selection(targets, eps_fraction, mode='max')
        elif type == 'lexicase':
            return lexicase_selection(targets, mode='max')
        elif type == 'rank_based':
            return rank_based(mode='max', pool_size=pool_size)
        elif type == 'roulette':
            return roulette_wheel_selection
        else:
            raise ValueError(f"Invalid selection type: {type}")
    else:
        raise ValueError(f"Invalid problem type: {problem}")


def tournament_selection_min(pool_size):
    """
    Returns a function that performs tournament selection to select an individual with the lowest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """

    def ts(pop):
        """
        Selects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness for ind in pool])]

    return ts


def tournament_selection_max(pool_size):
    """
    Returns a function that performs tournament selection to select an individual with the highest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """
    def ts(pop):
        """
        Selects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the highest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmax([ind.fitness for ind in pool])]

    return ts

def tournament_selection_min_size(pool_size, pressure_size=1e-4):
    """
    Returns a function that performs tournament selection to select an individual with the lowest fitness and size from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.
    pressure_size : float, optional
        Pressure for size in rank selection. Defaults to 1e-4.
    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the lowest fitness and size from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the combined lowest fitness and size in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """
    def ts(pop):
        """
        Selects the individual with the lowest fitness and size from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the combined lowest fitness and size in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness + pressure_size * ind.size for ind in pool])]

    return ts


def lexicase_selection(targets, mode='min'):
    """
    Returns a function that performs lexicase selection to select an individual with the lowest fitness
    from a population.

    Parameters
    ----------
    targets : torch.Tensor
        The true target values for each entry in the dataset (y_train).
    mode : str, optional
        The mode of selection. Can be 'min' or 'max'. Defaults to 'min'.

    Returns
    -------
    Callable
        A function ('ls') that performs lexicase selection on a population.
        
        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.

    Notes
    -----
    The returned function performs lexicase selection by receiving a population and returning the individual with the
    lowest fitness in the pool.
    """
    def ls(population):
        """
        Perform lexicase selection on a population of individuals.
        
        Parameters
        ----------
        population : list of Individual
            The population from which to select parents.

        Returns
        -------
        Individual
            The selected parent individual.
        """
        
        # Get errors for each individual on each test case
        errors = population.errors_case
        num_cases = errors.shape[1]  
                
        # Start with all individuals in the pool
        pool = population.population.copy()
        case_order = random.sample(range(num_cases), 5)
                        
        # Iterate over test cases and filter individuals based on exact performance (no epsilon)
        for i in range(5):
            # Generate an int from 0 to num_cases
            case_errors = errors[:, case_order[i]] ** 2
            
            # Get the best error on this test case across all individuals in the pool
            if mode == 'min':                
                best_individuals = np.where(case_errors == np.min(case_errors))[0]
            elif mode == 'max':
                best_individuals = np.where(case_errors == np.max(case_errors))[0]
                                                          
            # If only one individual remains, return it as the selected parent
            if len(best_individuals) == 1:
                return pool[best_individuals[0]]
            
            # Filter individuals based on exact performance and error
            pool = [pool[i] for i in best_individuals]
            errors = errors[best_individuals]
                    
        # If multiple individuals remain after all cases, return one at random
        return random.choice(pool)

    return ls  # Return the function that performs lexicase selection



# def epsilon_lexicase_selection(targets, eps_fraction=1e-7, mode='min'):
#     """
#     Returns a function that performs epsilon lexicase selection to select an individual with the lowest (or highest) fitness
#     from a population.

#     Parameters
#     ----------
#     targets : torch.Tensor
#         The true target values for each entry in the dataset (y_train).
#     eps_fraction : float, optional
#         The fraction of the population's standard deviation to use as the epsilon threshold. Defaults to 1e-7.
#     mode : str, optional
#         The mode of selection. Can be 'min' or 'max'. Defaults to 'min'.

#     Returns
#     -------
#     Callable
#         A selection function that, when given a population, returns the selected individual.
#     """
#     def els(pop):
#         """
#         Perform epsilon lexicase selection on a population of individuals.

#         Parameters
#         ----------
#         pop : Population
#             The population from which to select an individual. It is assumed that pop has attributes:
#             - population: list of Individual
#             - errors_case: a numpy array of shape (N, num_cases) with error values for each individual
#             - fit: numpy array of fitness values for each individual

#         Returns
#         -------
#         Individual
#             The selected individual.
#         """
#         errors = pop.errors_case            # shape: (N, num_cases)
#         fitness_values = pop.fit
#         fitness_std = np.std(fitness_values)  # Before was using the mean
#         epsilon = eps_fraction * fitness_std
#         num_cases = targets.shape[0]

#         # Start with all individuals represented by their indices.
#         candidate_indices = np.arange(len(pop.population))
#         case_order = random.sample(range(num_cases), 5)
        
#         for case_idx in case_order:
#             # Evaluate the squared error for current candidates on the selected case.
#             current_errors = errors[candidate_indices, case_idx] ** 2
            
#             if mode == 'min':
#                 best_value = np.min(current_errors)
#                 mask = current_errors <= best_value + epsilon
#             elif mode == 'max':
#                 best_value = np.max(current_errors)
#                 mask = current_errors >= best_value - epsilon
#             else:
#                 raise ValueError("Invalid mode. Use 'min' or 'max'.")
            
#             candidate_indices = candidate_indices[mask]
#             if candidate_indices.size == 1:
#                 return pop.population[candidate_indices[0]]
#             # If candidate_indices becomes empty (unlikely), break and select at random.
#             if candidate_indices.size == 0:
#                 break
        
#         # If multiple candidates remain, select one at random.
#         return pop.population[random.choice(candidate_indices.tolist())]
    
#     return els


def epsilon_lexicase_selection(targets, eps_fraction=1e-7, mode='min'):
    """
    Returns a function that performs epsilon lexicase selection to select an individual with the lowest fitness
    from a population.

    Parameters
    ----------
    targets : torch.Tensor
        The true target values for each entry in the dataset (y_train)
    eps_fraction : float, optional
        The fraction of the populations' standard deviation to use as the epsilon threshold. Defaults to 1e-6.
    mode : str, optional
        The mode of selection. Can be 'min' or 'max'. Defaults to 'min'.

    Returns
    -------
    Callable
        A function ('els') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.

    Notes
    -----
    The returned function performs lexicase selection by receiving a population and returning the individual with the
    lowest fitness in the pool.
    """

    def els(pop):
        """
        Perform epsilon lexicase selection on a population of individuals.

        Parameters
        ----------
        pop : list of Individual
            The population from which to select parents.
        targets : torch.Tensor
            The true target values for each entry in the dataset.
        epsilon : float, optional
            The epsilon threshold for lexicase selection. Defaults to 1e-6.

        Returns
        -------
        Individual
            The selected parent individual.
        """
        # Get errors for each individual on each test case
        errors = pop.errors_case
        fitness_values = pop.fit
        fitness_std = np.std(fitness_values)
        epsilon = eps_fraction * fitness_std
    
        num_cases = targets.shape[0]

        # Start with all individuals in the pool
        pool = pop.population.copy()
        case_order = random.sample(range(num_cases), 5)  # ADDED

        # Iterate over test cases and filter individuals based on epsilon threshold
        for i in range(5):
            case_errors = errors[:, case_order[i]] ** 2 

            # Get the best error on this test case across all individuals in the pool
            if mode == 'min':
                best_individuals = np.where(case_errors <= np.min(case_errors) + epsilon)[0]
            elif mode == 'max':
                best_individuals = np.where(case_errors >= np.max(case_errors) - epsilon)[0]

            # If only one individual remains, return it as the selected parent
            if len(best_individuals) == 1:
                return pool[best_individuals[0]]
            
            # Filter individuals based on epsilon threshold
            pool = [pool[i] for i in best_individuals]
            errors = errors[best_individuals]

        # If multiple individuals remain after all cases, return one at random
        return random.choice(pool)

    return els


def rank_based(mode='min', pool_size=2):
    """
    Returns a tournament function that performs rank-based selection to select an 
    individual with the lowest fitness and size from a population.

    Parameters
    ----------
    mode : str, optional
        The mode of selection. Can be 'min' or 'max'. Defaults to 'min'.
    pool_size : int, optional
        Number of individuals participating in the tournament. Defaults to 2.

    Returns
    -------
    Callable
        A function ('rs') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the combined lowest fitness and size in the pool.

    Notes
    -----
    The returned function performs rank-based selection by receiving a population and returning the individual with the
    lowest fitness 
    """

    if mode == 'max': 
        raise ValueError("Rank-based selection is only available for minimization problems.")

    def double_tournament(pop):
        """
        Perform rank-based selection on a population of individuals.

        Parameters
        ----------
        pop : list of Individual
            The population from which to select parents.

        Returns
        -------
        Individual
            The selected parent individual.
        """
        population, combined_ranks = pop.population, pop.combined_ranks
                
        # Randomly select `pool_size` individuals for the tournament
        selected_indices = np.random.choice(len(population), pool_size, replace=False)
        
        # Find the individual with the best combined rank in the pool
        best_index = min(selected_indices, key=lambda idx: combined_ranks[idx])
        
        return population[best_index]
    
    return double_tournament


def roulette_wheel_selection(population):
    """
    Perform roulette wheel selection on a population of individuals.

    Parameters
    ----------
    population : list of Individual
        The population from which to select parents.

    Returns
    -------
    Individual
        The selected parent individual.
    """
    # return random.choices(population, weights=[ind.fitness for ind in population])[0]
    return random.choices(population)[0]
 