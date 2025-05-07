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
SLIM_GSGP Class for Evolutionary Computation using PyTorch.
"""

import random
import time
import numpy as np
import torch

from slim_gsgp_lib_torch.algorithms.GP.representations.tree import Tree as GP_Tree
from slim_gsgp_lib_torch.algorithms.GSGP.representations.tree import Tree
from slim_gsgp_lib_torch.algorithms.SLIM_GSGP.representations.individual import Individual
from slim_gsgp_lib_torch.algorithms.SLIM_GSGP.representations.population import Population
from slim_gsgp_lib_torch.utils.diversity import gsgp_pop_div_from_vectors
from slim_gsgp_lib_torch.utils.logger import logger


class SLIM_GSGP:

    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        inflate_mutator,
        deflate_mutator,
        structure_mutator,
        xo_operator,
        verbose_reporter,
        ms,
        find_elit_func,
        p_xo=0.2,
        p_r=0.2,
        p_g=1,
        p_inflate=0.3,
        p_deflate=0.7,
        pop_size=100,
        p_prune=0.4,
        fitness_sharing=False,
        seed=0,
        operator="sum",
        struct_mutation=True,
        two_trees=True,
        p_struct_xo=0.5, 
        mut_xo_operator='rshuffle',
        settings_dict=None,
    ):
        """
        Initialize the SLIM_GSGP algorithm with given parameters.

        Parameters
        ----------
        pi_init : dict
            Dictionary with all the parameters needed for candidate solutions initialization.
        initializer : Callable
            Function to initialize the population.
        selector : Callable
            Function to select individuals.
        inflate_mutator : Callable
            Function for inflate mutation.
        deflate_mutator : Callable
            Function for deflate mutation.
        structure_mutator : Callable
            Function for structure mutation.
        verbose_reporter : Callable
            Function to report verbose information.
        xo_operator : Callable
            Crossover operator.
        ms : Callable
            Mutation step function.
        find_elit_func : Callable
            Function to find elite individuals.
        p_xo : float
            Probability of crossover. Default is 0.
        p_r : float
            Probability of replacing the GP tree. Default is 0.2.
        p_g : float
            Probability of grow mutation. Default is 1.
        p_inflate : float
            Probability of inflate mutation. Default is 0.3.
        p_deflate : float
            Probability of deflate mutation. Default is 0.7.
        p_prune : float
            Probability of prune mutation. Default is 0.4.
        pop_size : int
            Size of the population. Default is 100.
        fitness_sharing : bool
            Whether fitness sharing is used. Default is False.
        seed : int
            Random seed for reproducibility. Default is 0.
        operator : {'sum', 'prod'}
            Operator to apply to the semantics, either "sum" or "prod". Default is "sum".
        struct_mutation : bool
            Indicates if structure mutation is used. Default is True.
        two_trees : bool
            Indicates if two trees are used. Default is True.
        p_struct_xo : float
            Probability of structure crossover. Default is 0.5.
        mut_xo_operator : str
            Mutation operator for crossover. Default is 'rshuffle
        settings_dict : dict
            Additional settings passed as a dictionary.

        """
        self.pi_init = pi_init
        self.selector = selector
        self.p_inflate = p_inflate
        self.p_deflate = p_deflate
        self.inflate_mutator = inflate_mutator
        self.deflate_mutator = deflate_mutator
        self.structure_mutator = structure_mutator
        self.xo_operator = xo_operator
        self.ms = ms
        self.p_xo = p_xo
        self.p_r = p_r
        self.p_g = p_g
        self.initializer = initializer
        self.pop_size = pop_size
        self.p_prune = p_prune
        self.fitness_sharing = fitness_sharing
        self.seed = seed
        self.operator = operator
        self.struct_mutation = struct_mutation
        self.two_trees = two_trees
        self.settings_dict = settings_dict
        self.find_elit_func = find_elit_func
        self.p_struct_xo = p_struct_xo
        self.mut_xo_operator = mut_xo_operator
        self.verbose_reporter = verbose_reporter

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]

        GP_Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        GP_Tree.TERMINALS = pi_init["TERMINALS"]
        GP_Tree.CONSTANTS = pi_init["CONSTANTS"]

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        run_info,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        ffunction=None,
        max_depth=17,
        n_elites=1,
        reconstruct=True,
        n_jobs=1):
        """
        Solve the optimization problem using SLIM_GSGP.

        Parameters
        ----------
        X_train : array-like
            Training input data.
        X_test : array-like
            Testing input data.
        y_train : array-like
            Training output data.
        y_test : array-like
            Testing output data.
        curr_dataset : str or int
            Identifier for the current dataset.
        run_info : dict
            Information about the current run.
        n_iter : int
            Number of iterations. Default is 20.
        elitism : bool
            Whether elitism is used during evolution. Default is True.
        log : int or str
            Logging level (e.g., 0 for no logging, 1 for basic, etc.). Default is 0.
        verbose : int
            Verbosity level for logging outputs. Default is 0.
        test_elite : bool
            Whether elite individuals should be tested. Default is False.
        log_path : str
            File path for saving log outputs. Default is None.
        ffunction : function
            Fitness function used to evaluate individuals. Default is None.
        max_depth : int
            Maximum depth for the trees. Default is 17.
        n_elites : int
            Number of elite individuals to retain during selection. Default is True.
        reconstruct : bool
            Indicates if reconstruction of the solution is needed. Default is True.
        n_jobs : int
            Maximum number of concurrently running jobs for joblib parallelization. Default is 1.

        """

        if test_elite and (X_test is None or y_test is None):
            raise Exception('If test_elite is True you need to provide a test dataset')

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # starting time count
        start = time.time()

        # creating the initial population
        population = Population(
            [
                Individual(
                    collection=[
                        Tree(
                            tree,
                            train_semantics=None,
                            test_semantics=None,
                            reconstruct=True,
                        )
                    ],
                    train_semantics=None,
                    test_semantics=None,
                    reconstruct=True,
                )
                for tree in self.initializer(**self.pi_init)
            ]
        )

        # calculating initial population semantics
        population.calculate_semantics(X_train)

        # evaluating the initial population
        # ----------------- CHANGED -------------------
        population.evaluate(ffunction, y=y_train, operator=self.operator, n_jobs=n_jobs, fitness_sharing=self.fitness_sharing)
        # ----------------- END -------------------

        end = time.time()

        # setting up the elite(s)
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # calculating the testing semantics and the elite's testing fitness if test_elite is true
        if test_elite:
            population.calculate_semantics(X_test, testing=True)
            self.elite.evaluate(
                ffunction, y=y_test, testing=True, operator=self.operator
            )

        # logging the results based on the log level
        if log != 0:
            if log == 2:
                gen_diversity = (
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.sum(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        ),
                    )
                    if self.operator == "sum"
                    else gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.prod(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        )
                    )
                )
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes_count,
                    float(gen_diversity),
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
                gen_diversity = (
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.sum(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        ),
                    )
                    if self.operator == "sum"
                    else gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.prod(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        )
                    )
                )
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes_count,
                    float(gen_diversity),
                    np.std(population.fit),
                    " ".join([str(ind.nodes_count) for ind in population.population]),
                    " ".join([str(f) for f in population.fit]),
                    log,
                ]

            else:

                add_info = [self.elite.test_fitness, self.elite.nodes_count, log]

            logger(
                log_path,
                0,
                self.elite.fitness,
                end - start,
                float(population.nodes_count),
                additional_infos=add_info,
                run_info=run_info,
                seed=self.seed,
            )

        # displaying the results on console if verbose level is more than 0

        # ----------------- ADDED -------------------
        if self.operator == 'sum':
                pop_div_from_vectors = gsgp_pop_div_from_vectors(
                    torch.stack(
                        [torch.sum(ind.train_semantics, dim=0) for ind in population.population]
                    )
                )
        else:
            pop_div_from_vectors = gsgp_pop_div_from_vectors(
                torch.stack(
                    [torch.prod(ind.train_semantics, dim=0) for ind in population.population]
                )
            )

        # Check diversity of the first plot of the population 
        if reconstruct:
            div, seen = 0, []
            for individual in population.population:
                if individual.structure[0] not in seen:
                    seen.append(individual.structure[0])
                    div += 1
            div = div / len(population.population)
        
        # ----------------- END -------------------

        if verbose != 0:
            stats_data = {
                "dataset": curr_dataset,
                "iteration": 0,
                "train_fitness": self.elite.fitness,
                "test_fitness": self.elite.test_fitness,
                "time": end - start,
                "nodes": self.elite.nodes_count,
                "diversity": round(pop_div_from_vectors.item(),2),
                "avg_size": np.mean([ind.size for ind in population.population]),
                "avg_fitness": np.mean(population.fit),
                "std_fitness": np.std(population.fit),
            }

            self.verbose_reporter(
                stats_data,
                first=True,
                col_width=14,
            )


        # begining the evolution process
        for it in range(1, n_iter + 1, 1):
            # ----------------- ADDED -------------------
            # Check diversity of the first plot of the population 
            if reconstruct:
                div, seen = 0, []
                for individual in population.population:
                    if individual.structure[0] not in seen:
                        seen.append(individual.structure[0])
                        div += 1
                div = div / len(population.population)
            # ----------------- END -------------------


            # starting an empty offspring population
            offs_pop, start = [], time.time()

            # adding the elite to the offspring population, if applicable
            if elitism:
                offs_pop.extend(self.elites)

            # filling the offspring population
            while len(offs_pop) < self.pop_size:

                # Cross-over was selected
                if random.random() < self.p_xo:
                    p1, p2 = self.selector(population), self.selector(population)
                    while p1 == p2:
                        p1, p2 = self.selector(population), self.selector(population)
                    offs = self.xo_operator(p1, p2,
                                            X=X_train,
                                            X_test=X_test,
                                            reconstruct=reconstruct,)
                    offs_pop.extend(offs)

                # Mutation was selected
                else:
                    p1 = self.selector(population)

                    # so, mutation was selected. Now deflation or inflation is selected.
                    if random.random() < self.p_deflate:
                        # if the parent has only one block, it cannot be deflated
                        if p1.size == 1:
                            # if structure_mutation is set to true, the parent who cannot be deflated will be structurally mutated
                            if self.struct_mutation:
                                off1 = self.structure_mutator(individual=p1, 
                                                          X=X_train,
                                                          max_depth=self.pi_init["init_depth"],
                                                          p_c=self.pi_init["p_c"],
                                                          X_test=X_test,
                                                          grow_probability=self.p_g, 
                                                          replace_probability=self.p_r,
                                                          p_prune = self.p_prune,
                                                          reconstruct=reconstruct)

                            else:
                                # if we choose to not structure mutate the parent, we inflate it instead
                                ms_ = self.ms()
                                off1 = self.inflate_mutator(
                                    p1,
                                    ms_,
                                    X_train,
                                    max_depth=self.pi_init["init_depth"],
                                    p_c=self.pi_init["p_c"],
                                    X_test=X_test,
                                    reconstruct=reconstruct,
                                    grow_probability=self.p_g,
                                )

                        else:
                            # if the size of the parent is more than 1, normal deflation can occur
                            off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                    # inflation mutation was selected
                    else:
                        # selecting a parent to inflate
                        p1 = self.selector(population)

                        # determining the random mutation step
                        ms_ = self.ms()

                        # if the chosen parent is already at maximum depth and therefore cannot be inflated
                        if max_depth is not None and p1.depth == max_depth:
                            # if structure mutation is set to true, the parent who cannot be inflated will be structurally mutated
                            if self.struct_mutation:                                
                                # Not structure mutating the individual, as it is fully evolved and mutation could be proven to be harmful
                                off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                            # if copy parent is false, the parent is deflated instead of inflated
                            else:
                                off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                        # so the chosen individual can be normally inflated
                        else:
                            off1 = self.inflate_mutator(
                                p1,
                                ms_,
                                X_train,
                                max_depth=self.pi_init["init_depth"],
                                p_c=self.pi_init["p_c"],
                                X_test=X_test,
                                reconstruct=reconstruct,
                                grow_probability=self.p_g,
                            )

                        # if offspring resulting from inflation exceedes the max depth
                        if max_depth is not None and off1.depth > max_depth:
                            # if structure mutation is set to true, the offspring is discarded and the parent is structurally mutated
                            if self.struct_mutation:
                                off1 = self.deflate_mutator(p1, reconstruct=reconstruct)
                                
                            else:
                                # otherwise, deflate the parent
                                off1 = self.deflate_mutator(p1, reconstruct=reconstruct)
                            
                    # adding the new offspring to the offspring population
                    offs_pop.append(off1)

            # removing any excess individuals from the offspring population
            if len(offs_pop) > population.size:
                offs_pop = offs_pop[: population.size]

            # turning the offspring population into a Population
            offs_pop = Population(offs_pop)
            # calculating the offspring population semantics
            offs_pop.calculate_semantics(X_train)

            # ----------------- CHANGED -------------------
            # evaluating the offspring population
            offs_pop.evaluate(ffunction, y=y_train, operator=self.operator, n_jobs=n_jobs, fitness_sharing=self.fitness_sharing)
            # ----------------- END -------------------

            # replacing the current population with the offspring population P = P'
            population = offs_pop
            self.population = population

            end = time.time()

            # setting the new elite(s)
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            # calculating the testing semantics and the elite's testing fitness if test_elite is true
            if test_elite:
                self.elite.calculate_semantics(X_test, testing=True)
                self.elite.evaluate(
                    ffunction, y=y_test, testing=True, operator=self.operator
                )


            # ----------------- ADDED -------------------
            if self.operator == 'sum':
                pop_div_from_vectors = gsgp_pop_div_from_vectors(
                    torch.stack(
                        [torch.sum(ind.train_semantics, dim=0) for ind in population.population]
                    )
                )
            else:
                pop_div_from_vectors = gsgp_pop_div_from_vectors(
                    torch.stack(
                        [torch.prod(ind.train_semantics, dim=0) for ind in population.population]
                    )
                )

            # displaying the results on console if verbose level is more than 0
            if verbose != 0:
                stats_data = {
                    "dataset": curr_dataset,
                    "iteration": it,
                    "train_fitness": self.elite.fitness,
                    "test_fitness": self.elite.test_fitness,
                    "time": end - start,
                    "nodes": self.elite.nodes_count,
                    "diversity": round(pop_div_from_vectors.item(),2),
                    "avg_size": np.mean([ind.size for ind in population.population]),
                    "avg_fitness": np.mean(population.fit),
                    "std_fitness": np.std(population.fit),
                }

                self.verbose_reporter(
                    stats_data,
                    col_width=14,
                )
            # ----------------- END -------------------

            # logging the results based on the log level
            if log != 0:

                if log == 2:
                    gen_diversity = (
                        pop_div_from_vectors
                    )
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes_count,
                        float(gen_diversity),
                        np.std(population.fit),
                        log,
                    ]

                elif log == 3:
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes_count,
                        " ".join(
                            [str(ind.nodes_count) for ind in population.population]
                        ),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                elif log == 4:
                    gen_diversity = (
                        gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.sum(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            ),
                        )
                        if self.operator == "sum"
                        else gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.prod(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            )
                        )
                    )
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes_count,
                        float(gen_diversity),
                        np.std(population.fit),
                        " ".join(
                            [str(ind.nodes_count) for ind in population.population]
                        ),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                else:
                    add_info = [self.elite.test_fitness, self.elite.nodes_count, log]

                logger(
                    log_path,
                    it,
                    self.elite.fitness,
                    end - start,
                    float(population.nodes_count),
                    additional_infos=add_info,
                    run_info=run_info,
                    seed=self.seed,
                )