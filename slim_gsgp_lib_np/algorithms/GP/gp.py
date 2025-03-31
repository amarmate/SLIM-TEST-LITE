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
Genetic Programming (GP) module.
"""

import random
import time

import numpy as np
from slim_gsgp_lib_np.algorithms.GP.representations.population import Population
from slim_gsgp_lib_np.algorithms.GP.representations.tree import Tree
from slim_gsgp_lib_np.utils.diversity import niche_entropy
from slim_gsgp_lib_np.utils.logger import logger
from slim_gsgp_lib_np.utils.diversity import gsgp_pop_div_from_vectors
from slim_gsgp_lib_np.utils.utils import verbose_reporter

class GP:
    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        mutator,
        crossover,
        find_elit_func,
        p_m=0.2,
        p_xo=0.8,
        pop_size=100,
        seed=0,
        settings_dict=None,
        callbacks=None, 
    ):
        """
        Initialize the Genetic Programming algorithm.

        Parameters
        ----------
        pi_init : dict
            Dictionary with all the parameters needed for candidate solutions initialization.
        initializer : Callable
            Function to initialize the population.
        selector : Callable
            Function to select individuals.
        mutator : Callable
            Function to mutate individuals.
        crossover : Callable
            Function to perform crossover between individuals.
        find_elit_func : Callable
            Function to find elite individuals.
        p_m : float, optional
            Probability of mutation. Default is 0.2.
        p_xo : float, optional
            Probability of crossover. Default is 0.8.
        pop_size : int, optional
            Size of the population. Default is 100.
        seed : int, optional
            Seed for random number generation. Default is 0.
        settings_dict : dict, optional
            Additional settings dictionary.
        callbacks : list, optional
            List of callbacks to be executed during the evolutionary process
        """
        self.pi_init = pi_init
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.mutator = mutator
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        self.find_elit_func = find_elit_func
        self.settings_dict = settings_dict
        self.callbacks = callbacks if callbacks is not None else []

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        run_info=None,
        max_depth=None,
        ffunction=None,
        n_elites=1,
        depth_calculator=None,
        n_jobs=1
    ):
        """
        Execute the Genetic Programming algorithm.
        """
        # Set seeds for reproducibility.
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        # Initialize the population.
        population = Population([Tree(tree) for tree in self.initializer(**self.pi_init)])
        population.calculate_semantics(X_train, testing=False)
        population.calculate_errors_case(y_train)
        population.evaluate(target=y_train, testing=False)
        self.population = population
        self.dataset = curr_dataset

        end = time.time()
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # Test the elite on the testing data if required.
        if test_elite:
            self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

        # Logging initial generation.
        self.log_generation(0, population, end - start, log, log_path, run_info) if log != 0 else None

        # Initialize the time tracker for mutation and crossover.
        self.time_dict = {'mutation': [], 'xo': []}
        if self.selector.__name__ in ["els", "mels"]:
            self.lex_rounds = [0]

        # Verbose reporting.
        self.print_results(0, start, end) if verbose != 0 else None

        for callback in self.callbacks:
            callback.on_train_start(self)

        # ------------------------- EVOLUTIONARY PROCESS -------------------------
        for it in range(1, n_iter + 1):
            self.lex_rounds = [] if self.selector.__name__ in ["els", "mels"] else None

            # Reset the timing dictionary for the new generation.
            self.time_dict = {'mutation': [], 'xo': []}

            for callback in self.callbacks:
                callback.on_generation_start(self, it)

            # Evolve the population for this generation.
            offs_pop, gen_start = self.evolve_population(
                population,
                ffunction,
                max_depth,
                depth_calculator,
                elitism,
                X_train,
                y_train,
            )
            population = offs_pop
            gen_end = time.time()

            # Get the new elite(s) and test on test data if required.
            self.elites, self.elite = self.find_elit_func(population, n_elites)
            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

            # Logging the generation and verbose reporting.
            self.log_generation(it, population, gen_end - gen_start, log, log_path, run_info) if log != 0 else None
            self.print_results(it, gen_start, gen_end) if verbose != 0 else None

            for callback in self.callbacks:
                callback.on_generation_end(self, it, gen_start, gen_end)
        
        for callback in self.callbacks:
            callback.on_train_end(self)

    def evolve_population(
        self,
        population,
        ffunction,
        max_depth,
        depth_calculator,
        elitism,
        X_train,
        y_train,
    ):
        """
        Evolve the population for one generation.
        """
        offs_pop = []
        gen_start = time.time()

        # Add elites if elitism is enabled.
        if elitism:
            offs_pop.extend(self.elites)

        # Fill the offspring population.
        while len(offs_pop) < self.pop_size:
            # Randomly decide whether to use crossover or mutation.
            if random.random() < self.p_xo:
                offspring = self.crossover_step(population, max_depth, depth_calculator)
            else:
                offspring = [self.mutation_step(population, max_depth, depth_calculator)]
            offs_pop.extend([Tree(child) for child in offspring])

        # Ensure the offspring population matches the required size.
        if len(offs_pop) > population.size:
            offs_pop = offs_pop[: population.size]

        # Evaluate the new population.
        offs_pop = Population(offs_pop)
        offs_pop.calculate_semantics(X_train, testing=False)
        offs_pop.calculate_errors_case(y_train)
        offs_pop.evaluate(target=y_train, testing=False)
        self.population = offs_pop

        return offs_pop, gen_start

    def crossover_step(self, population, max_depth, depth_calculator):
        """
        Perform the crossover operation while tracking its execution time.
        """
        start = time.time()

        # Select two distinct parents.
        p1 = self.selector(population)
        p2 = self.selector(population)
        while p1 == p2:
            p1 = self.selector(population)
            p2 = self.selector(population)

        if self.selector.__name__ in ["els", "mels"]:
            p1, i1 = p1
            p2, i2 = p2
            self.lex_rounds.extend([i1, i2])

        # Generate offspring from the selected parents.
        offs1, offs2 = self.crossover(
            p1.repr_,
            p2.repr_,
            tree1_n_nodes=p1.nodes_count,
            tree2_n_nodes=p2.nodes_count,
        )

        # Ensure the offspring do not exceed the maximum allowed depth.
        if max_depth is not None:
            while depth_calculator(offs1) > max_depth or depth_calculator(offs2) > max_depth:
                offs1, offs2 = self.crossover(
                    p1.repr_,
                    p2.repr_,
                    tree1_n_nodes=p1.nodes_count,
                    tree2_n_nodes=p2.nodes_count,
                )

        elapsed = time.time() - start
        self.time_dict['xo'].append(elapsed)
        return [offs1, offs2]

    def mutation_step(self, population, max_depth, depth_calculator):
        """
        Perform the mutation operation while tracking its execution time.
        """
        start = time.time()

        # Select a parent and mutate.
        p1 = self.selector(population) 

        if self.selector.__name__ in ["els", "mels"]:
            p1, i1 = p1
            self.lex_rounds.append(i1)

        offs1 = self.mutator(p1.repr_, num_of_nodes=p1.nodes_count)

        # Ensure the mutated offspring does not exceed the maximum allowed depth.
        if max_depth is not None:
            while depth_calculator(offs1) > max_depth:
                offs1 = self.mutator(p1.repr_, num_of_nodes=p1.nodes_count)

        elapsed = time.time() - start
        self.time_dict['mutation'].append(elapsed)
        return offs1
    
    def log_generation(self, generation, population, elapsed_time, log, log_path, run_info):
        """
        Log the results for the current generation including mutation and crossover timings.
        """
        # Prepare additional logging info based on the log level.
        if log == 2:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                log,
            ]
        elif log == 3:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                " ".join([str(ind.nodes_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        elif log == 4:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                " ".join([str(ind.nodes_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        else:
            add_info = [self.elite.test_fitness, self.elite.nodes_count, log]

        logger(
            log_path,
            generation,
            self.elite.fitness,
            elapsed_time,
            float(population.nodes_count),
            additional_infos=add_info,
            run_info=run_info,
            seed=self.seed,
        )

    def print_results(self, iteration, start, end):
        params = {
            "dataset": self.dataset,
            "it": iteration,
            "train": self.elite.fitness,
            "test": self.elite.test_fitness,
            "time": end - start,
            "nodes": self.elite.nodes_count,
            "avg_nodes": np.mean([ind.nodes_count for ind in self.population.population]),
            "div": int(self.calculate_diversity()),
            "mut": f"{np.round(1000*np.mean([self.time_dict['mutation']]),2) if self.time_dict['mutation'] != [] else 'N/A'} ({len(self.time_dict['mutation'])})",
            "xo": f"{np.round(1000*np.mean([self.time_dict['xo']]),2) if self.time_dict['xo'] != [] else 'N/A'} ({len(self.time_dict['xo'])})",
        }
        if self.selector.__name__ in ["els", "mels"]:
            params["lex_r"] = np.mean(self.lex_rounds)
        
        verbose_reporter(
                params, 
                first = iteration == 0,
                precision=3, 
                col_width=14
        )

    def calculate_diversity(self):
        return gsgp_pop_div_from_vectors(self.population.train_semantics)



