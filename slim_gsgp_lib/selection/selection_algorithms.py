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
        The type of selection to perform. Can be 'tournament', 'e_lexicase' or 'lexicase. Defaults to 'tournament'.
    pool_size : int, optional
        Number of individuals participating in the tournament. Defaults to 2.
    eps_fraction : float, optional
        The fraction of the populations' standard deviation to use as the epsilon threshold. Defaults
    targets : torch.Tensor, optional
        The true target values for each entry in the dataset. Required for lexicase selection and epsilon lexicase

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
        else:
            raise ValueError(f"Invalid selection type: {type}")
    elif problem == 'max':
        if type == 'tournament':
            return tournament_selection_max(pool_size)
        elif type == 'e_lexicase':
            return epsilon_lexicase_selection(targets, eps_fraction, mode='max')
        elif type == 'lexicase':
            return lexicase_selection(targets, mode='max')
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


def epsilon_lexicase_selection(targets, eps_fraction=1e-4, mode='min'):
    """
    Returns a function that performs epsilon lexicase selection to select an individual with the lowest fitness
    from a population.

    Parameters
    ----------
    targets : torch.Tensor
        The true target values for each entry in the dataset (y_train)
    eps_fraction : float, optional
        The fraction of the populations' standard deviation to use as the epsilon threshold. Defaults to 1e-4.
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
        errors = [ind.evaluate_per_case(targets) for ind in pop.population]
        fitness_values = [ind.fitness for ind in pop.population]
        fitness_std = np.std(fitness_values)
        epsilon = eps_fraction * fitness_std
        
        # Randomly shuffle the order of test cases
        num_cases = targets.shape[0]
        case_order = list(range(num_cases))
        random.shuffle(case_order)
        
        # Start with all individuals in the pool
        pool = pop.copy()
        
        # Iterate over test cases and filter individuals based on epsilon threshold
        for case_idx in case_order:
            # Get the best error on this test case across all individuals in the pool
            if mode == 'min':
                best_error = min(errors[i][case_idx] for i, ind in enumerate(pool))
                pool = [ind for i, ind in enumerate(pool) if errors[i][case_idx] <= best_error + epsilon]
            elif mode == 'max':
                best_error = max(errors[i][case_idx] for i, ind in enumerate(pool))
                pool = [ind for i, ind in enumerate(pool) if errors[i][case_idx] >= best_error - epsilon]
            
            # If only one individual remains, return it as the selected parent
            if len(pool) == 1:
                return pool[0]
        
        # If multiple individuals remain after all cases, return one at random
        return random.choice(pool)

    return els


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
        errors = [ind.error_per_case(targets) for ind in population.population]
        
        # Randomly shuffle the order of test cases
        num_cases = targets.shape[0]
        case_order = list(range(num_cases))
        random.shuffle(case_order)
        
        # Start with all individuals in the pool
        pool = population.population.copy()
        
        # Iterate over test cases and filter individuals based on exact performance (no epsilon)
        for case_idx in case_order:
            # Get the best error on this test case across all individuals in the pool
            if mode == 'min':
                best_error = min(errors[i][case_idx] for i, ind in enumerate(pool))
            elif mode == 'max':
                best_error = max(errors[i][case_idx] for i, ind in enumerate(pool))
            
            # Filter out individuals whose error exceeds best_error (strict comparison)
            pool = [ind for i, ind in enumerate(pool) if errors[i][case_idx] == best_error]
            
            # If only one individual remains, return it as the selected parent
            if len(pool) == 1:
                return pool[0]
        
        # If multiple individuals remain after all cases, return one at random
        return random.choice(pool)

    return ls  # Return the function that performs lexicase selection
