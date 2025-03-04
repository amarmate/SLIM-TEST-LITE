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
Population class implementation for evaluating genetic programming trees.
"""
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree_utils import _execute_tree
import numpy as np

class Population:
    """
    The Population class representing a population of trees in MULTI-SLIM-GSGP.
    
    Attributes
    ----------
    population : List
        List of Tree objects representing the population.   
    size : int
        Number of trees in the population.
    nodes_count : int
        Total number of nodes in the population.
    fit : np.ndarray
        Fitness values of the population.
    train_semantics : np.ndarray
        Training semantics of the population.
    test_semantics : np.ndarray
        Testing semantics of the population.

    Methods
    -------
    calculate_semantics(inputs, testing=False)
        Calculate the semantics for each tree in the population.
    evaluate(target, testing=False) 
        Evaluate the population using the errors per case with MSE.
    __len__()
        Return the size of the population.
    __getitem__(item)
        Get an individual from the population by index.
        
    """

    def __init__(self, pop):
        """
        Initializes a population of Trees.

        This constructor sets up the population with a list of Tree objects,
        calculating the size of the population and the total node count.

        Parameters
        ----------
        pop : List
            The list of tree Tree objects that make up the population.

        Returns
        -------
        None
        """
        self.population = pop
        self.size = len(pop)
        self.nodes_count = sum(ind.nodes_count for ind in pop)
        self.fit = None

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for each tree in the population.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for calculating semantics.
        testing : bool, optional
            Boolean indicating if the calculation is for testing semantics.

        Returns
        -------
        None
        """
        # computing the semantics for all the trees in the population
        if testing and hasattr(self, "test_semantics"):
            print("Warning: Testing semantics already calculated.")
            return
        if not testing and hasattr(self, "train_semantics"):
            print("Warning: Training semantics already calculated.")
            return
        
        [
            tree.calculate_semantics(inputs, testing)
            for tree in self.population
        ]

        # computing testing semantics, if applicable
        if testing:
            # setting the population semantics to be a list with all the semantics of all trees
            self.test_semantics = np.array([
                tree.test_semantics for tree in self.population
            ])

        else:
            # setting the population semantics to be a list with all the semantics of all trees
            self.train_semantics = np.array([
                tree.train_semantics for tree in self.population
            ])


    def evaluate(self, target, testing=False):
        """
        Evaluate the population using the errors per case with MSE

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the individuals.
        target : torch.Tensor        
            Expected output (target) values.

        Returns
        -------
        None
        """
        if testing and not hasattr(self, "test_semantics"):
            raise ValueError("Testing semantics not calculated.")
        
        elif not testing and not hasattr(self, "train_semantics"):
            raise ValueError("Training semantics not calculated.")

        # Check if errors case is already calculated
        sem = self.test_semantics if testing else self.train_semantics
        errors = sem - np.stack([target] * sem.shape[0])
        fitness = np.sqrt(np.mean(errors**2, axis=1))
        self.fit = fitness
        for i, individual in enumerate(self.population):
            individual.fitness = fitness[i]
            
        
    def __len__(self):
        """
        Return the size of the population.

        Returns
        -------
        int
            Size of the population.
        """
        return self.size

    def __getitem__(self, item):
        """
        Get an individual from the population by index.

        Parameters
        ----------
        item : int
            Index of the individual to retrieve.

        Returns
        -------
        Individual
            The individual at the specified index.
        """
        return self.population[item]
