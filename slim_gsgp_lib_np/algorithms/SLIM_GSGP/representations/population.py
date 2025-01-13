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
from slim_gsgp_lib_np.utils.utils import _evaluate_slim_individual
from joblib import Parallel, delayed
import numpy as np


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
        self.errors_case = None
        self.sizes = [ind.nodes_count for ind in population]

    def set_unique_id(self): 
        """
        Set a unique id for each individual in the population.

        Returns
        -------
        None
        """
        for i, individual in enumerate(self.population):
            individual.__setattr__('id', i)

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
            
    def calculate_errors_case(self, target, operator="sum"):
        """
        Calculate the errors case for each individual in the population.

        Parameters
        ----------
        y_train : torch.Tensor
            Expected output (target) values for training.

        Returns
        -------
        None
        """
        # computing the errors case for all the individuals in the population
        [
            individual.calculate_errors_case(target, operator=operator)
            for individual in self.population
        ]

        # defining the errors case of the population to be a list with the errors case of all individuals in the population
        self.errors_case = np.array([individual.errors_case for individual in self.population])

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
    
    def evaluate(self, 
            ffunction, 
            y, 
            operator="sum", 
            n_jobs=1, 
            fitness_sharing=False,
            rank_selection=False, 
            pressure_size=0.5,
            ):
        
        if n_jobs == 1 and not rank_selection:
            self.evaluate_no_parall(ffunction, y, operator)
        else:
            self.evaluate_parall(
                ffunction, y, operator, n_jobs, fitness_sharing, rank_selection, pressure_size
            )

    def evaluate_no_parall(self, ffunction, y, operator="sum"):
        """
        Evaluate the population using a fitness function (without parallelization).
        This function is not currently in use, but has been retained for potential future use
        at the developer's discretion.

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the individuals.
        y : torch.Tensor
            Expected output (target) values.
        operator : str, optional
            Operator to apply to the semantics. Default is "sum".

        Returns
        -------
        None
        """
        # evaluating all the individuals in the population
        [
            individual.evaluate(ffunction, y, operator=operator)
            for individual in self.population
        ]
        # defining the fitness of the population to be a list with the fitnesses of all individuals in the population
        self.fit = [individual.fitness for individual in self.population]

    def evaluate_parall(self, 
                ffunction, 
                y, 
                operator="sum", 
                n_jobs=1, 
                rank_selection=False, 
                pressure_size=0.5,
                ):
        """
        Evaluate the population using a fitness function and calculate ranks for fitness and size.

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the individuals.
        y : torch.Tensor
            Expected output (target) values.
        operator : str, optional
            Operator to apply to the semantics ("sum" or "prod"). Default is "sum".
        n_jobs : int, optional
            The maximum number of concurrently running jobs for joblib parallelization. Default is 1.
        rank_selection : bool, optional
            Boolean indicating if rank selection is used. Default is False.
        pressure_size : float, optional
            Pressure for size in rank selection. Default is 0.5.

        Returns
        -------
        None
        """
        # Evaluate individuals' fitnesses
        self.fit = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_slim_individual)(
                individual, ffunction=ffunction, y=y, operator=operator
            ) for individual in self.population
        )
        
        [self.population[i].__setattr__('fitness', f) for i, f in enumerate(self.fit)]
        

        # Sort the population based on fitness and optionally calculate ranks
        if rank_selection:
            # Sorting the population 
            individuals_with_fitness = list(zip(self.population, self.fit, self.sizes))
            individuals_with_fitness.sort(key=lambda x: x[1])
            self.population, self.fit, self.sizes = zip(*individuals_with_fitness)
            self.population = list(self.population)
            self.fit = list(self.fit)
            self.sizes = list(self.sizes)
            
            # Calculate ranks
            fitness_ranks = list(range(len(self.population)))
            size_ranking_indices = np.argsort(self.sizes)
            size_ranks = np.empty(len(self.sizes), dtype=int)
            size_ranks[size_ranking_indices] = np.arange(len(self.sizes))

            self.combined_ranks = [
                fitness_rank + pressure_size * size_rank
                for fitness_rank, size_rank in zip(fitness_ranks, size_ranks)
            ]

            # # ----------------------- CHANGED ---------------------------
            # if fitness_sharing: 
            #     elite_id, elite_fit = np.argmin(self.fit), np.min(self.fit)     
            #     seen = {}
            #     for individual in self.population:
            #         if individual.structure[0] not in seen:
            #             seen[individual.structure[0]] = 1
            #         else:
            #             seen[individual.structure[0]] += 1
                
            #     for i, individual in enumerate(self.population):
            #         # individual.fitness = self.fit[i] * (np.log(seen[individual.structure[0]]+1))
            #         self.fit[i] = self.fit[i] * (np.log(seen[individual.structure[0]]+10))

            # ----------------------- END ---------------------------

                # Assigning individuals' fitness as an attribute
                        
                
            

