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
Population Class for SLIM GSGP using PyTorch.
"""
from slim_gsgp_lib.utils.utils import _evaluate_slim_individual
from joblib import Parallel, delayed
import torch 

class Population:
    def __init__(self, population):
        """
        Initialize the Population with a list of individuals.

        Parameters
        ----------
        population : list
            The list of individuals in the population.

        Returns
        -------
        None
        """
        self.population = population
        self.size = len(population)
        self.nodes_count = sum([ind.nodes_count for ind in population])
        self.fit = None
        self.train_semantics = None
        self.test_semantics = None

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for each individual in the population.

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
        # computing the semantics for all the individuals in the population
        [
            individual.calculate_semantics(inputs, testing)
            for individual in self.population
        ]

        # computing testing semantics, if applicable
        if testing:
            # setting the population semantics to be a list with all the semantics of all individuals
            self.test_semantics = [
                individual.test_semantics for individual in self.population
            ]

        else:
            # setting the population semantics to be a list with all the semantics of all individuals
            self.train_semantics = [
                individual.train_semantics for individual in self.population
            ]

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

    # def evaluate(self, ffunction, y, operator="sum", **kwargs):
    #     """
    #     Evaluate the population using a fitness function (without parallelization).
    #     This function is not currently in use, but has been retained for potential future use
    #     at the developer's discretion.

    #     Parameters
    #     ----------
    #     ffunction : Callable
    #         Fitness function to evaluate the individuals.
    #     y : torch.Tensor
    #         Expected output (target) values.
    #     operator : str, optional
    #         Operator to apply to the semantics. Default is "sum".

    #     Returns
    #     -------
    #     None
    #     """
    #     # evaluating all the individuals in the population
    #     [
    #         individual.evaluate(ffunction, y, operator=operator)
    #         for individual in self.population
    #     ]
    #     # defining the fitness of the population to be a list with the fitnesses of all individuals in the population
    #     self.fit = [individual.fitness for individual in self.population]

    def evaluate(self, ffunction, y, operator="sum", **kwargs):
        """
        Evaluate the population using a matrix-based approach with torch.sub for difference calculation.
        
        Parameters
        ----------
        ffunction : Callable
            A fitness function (e.g., RMSE) that expects target and predictions.
        y : torch.Tensor
            Target values of shape (1, n) (e.g., (1, 224)).
        operator : str, optional
            Operator to apply to the semantics (not used in this RMSE example). Default is "sum".
        
        Returns
        -------
        None
        """
        # Normalize the semantics for each individual so they all have shape (1, n)
        if isinstance(self.train_semantics, list):
            normalized_semantics = []
            for sem in self.train_semantics:
                # Ensure sem is a tensor
                if not isinstance(sem, torch.Tensor):
                    sem = torch.tensor(sem)
                # If sem is 1D, make it 2D (1, n)
                if sem.ndim == 1:
                    sem = sem.unsqueeze(0)
                # If sem has more than one row (e.g., (2, n)), average over the rows to get (1, n)
                elif sem.ndim > 1 and sem.shape[0] != 1:
                    sem = sem.mean(dim=0, keepdim=True)
                normalized_semantics.append(sem)
            # Stack them along a new dimension (resulting in shape: (pop_size, 1, n))
            S = torch.stack(normalized_semantics)
            # Squeeze out the singleton dimension to get shape: (pop_size, n)
            S = S.squeeze(1)
        else:
            S = self.train_semantics
            if S.ndim == 3 and S.shape[1] != 1:
                S = S.mean(dim=1)
        
        # Now S has shape (pop_size, n)
        # Compute the difference using broadcasting: y is (1, n) and S is (pop_size, n)
        diff = torch.sub(y, S)
        # Compute squared differences
        squared_diff = torch.square(diff)
        # Compute the mean squared error for each individual across the output dimension
        mse = torch.mean(squared_diff, dim=1)
        # Compute RMSE for each individual (resulting in shape: (pop_size,))
        fitness = torch.sqrt(mse)
        
        # Store the computed fitness values in self.fit
        self.fit = fitness
        
        # Update each individual's fitness attribute by converting each tensor to a scalar
        for i, individual in enumerate(self.population):
            individual.fitness = self.fit[i].item()


    # def evaluate(self, ffunction, y, operator="sum", n_jobs=1):
    #     """
    #     Evaluate the population using a fitness function.

    #     Parameters
    #     ----------
    #     ffunction : Callable
    #         Fitness function to evaluate the individuals.
    #     y : torch.Tensor
    #         Expected output (target) values.
    #     operator : str, optional
    #         Operator to apply to the semantics ("sum" or "prod"). Default is "sum".
    #     n_jobs : int, optional
    #         The maximum number of concurrently running jobs for joblib parallelization. Default is 1.

    #     Returns
    #     -------
    #     None
    #     """
    #     # Evaluates individuals' fitnesses
    #     self.fit = Parallel(n_jobs=n_jobs)(
    #         delayed(_evaluate_slim_individual)(individual, ffunction=ffunction, y=y, operator=operator
    #         ) for individual in self.population)

    #     # Assigning individuals' fitness as an attribute
    #     [self.population[i].__setattr__('fitness', f) for i, f in enumerate(self.fit)]

