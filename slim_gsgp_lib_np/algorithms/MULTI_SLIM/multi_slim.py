from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree import Tree
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.population import Population    
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.condition import Condition
import random 
import time
import numpy as np

from slim_gsgp_lib_np.utils.logger import logger
from slim_gsgp_lib_np.utils.utils import verbose_reporter
from slim_gsgp_lib_np.utils.diversity import gsgp_pop_div_from_vectors

class MULTI_SLIM:
    def __init__(self, pi_init, mutator, xo_operator, selector, initializer, find_elit_func,
                 p_mut, p_xo, seed, callbacks, decay_rate):
        """
        Initialize the MULTI_SLIM optimizer.

        Parameters
        ----------
        pi_init : dict
            Dictionary containing initial parameters, including:
              - FUNCTIONS, TERMINALS, CONSTANTS for tree representation.
              - SPECIALISTS (a dict mapping specialist names to individuals).
              - pop_size, depth_condition, max_depth, etc.
        mutator : Callable
            Mutation operator (aggregated) for modifying individuals.
        xo_operator : Callable 
            Crossover operator for modifying individuals 
        selector : Callable
            Selection operator to choose individuals from the population.
        initializer : Callable
            Function that initializes a population of individuals.
        find_elit_func : Callable
            Function to identify the best individual in a population.
        p_mut : float
            Mutation probability.
        p_xo : float
            Crossover probability.
        seed : int
            Seed for the random number generator.
        callbacks : list
            List of callback functions to be invoked each iteration.
        decay_rate : float
            Decay rate parameter used in some mutation operations.
        """
        # Save parameters
        self.pi_init = pi_init
        self.pop_size = pi_init["pop_size"]
        self.mutator = mutator
        self.xo_operator = xo_operator 
        self.selector = selector
        self.initializer = initializer
        self.find_elit_func = find_elit_func
        self.p_mut = p_mut
        self.p_xo = p_xo
        self.seed = seed
        self.callbacks = callbacks if callbacks is not None else []
        self.decay_rate = decay_rate
        self.stop_training = False

        # Set tree representation parameters
        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]
        Tree.SPECIALISTS = pi_init["SPECIALISTS"]

        Condition.FUNCTIONS = pi_init["FUNCTIONS"]
        Condition.TERMINALS = pi_init["TERMINALS"]
        Condition.CONSTANTS = pi_init["CONSTANTS"]

    def solve(self, X_train, X_test, y_train, y_test, curr_dataset,
              run_info, ffunction, log, verbose, n_iter, test_elite,
              log_path, n_elites, elitism, timeout, **kwargs):
        """
        Run the MULTI_SLIM evolutionary algorithm.

        Parameters
        ----------
        X_train : np.Tensor or similar
            Training input data.
        X_test : np.Tensor or similar
            Testing input data.
        y_train : np.Tensor or similar
            Training target values.
        y_test : np.Tensor or similar
            Testing target values.
        curr_dataset : str
            Name of the current dataset.
        run_info : list
            Run information (e.g., [ALGORITHM, slim_version, UNIQUE_RUN_ID, dataset_name]).
        ffunction : Callable
            Fitness function that evaluates an individual.
        log : str
            Logging mode (e.g., 'min' or 'max').
        verbose : bool
            If True, prints progress information.
        n_iter : int
            Number of iterations (generations).
        test_elite : bool
            Whether to evaluate the elite on test data.
        log_path : str
            Path for logging.
        n_elites : int
            Number of elite individuals to preserve.
        elitism : bool
            Whether elitism is enabled.
        timeout : float
            Maximum allowed run time.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        best_ind : object
            The best individual found after evolution.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Check if test data is provided when test_elite is True.
        if test_elite and (X_test is None or y_test is None):
            raise Exception('If test_elite is True you need to provide a test dataset')

        # Initialize the population 
        start = time.time()
        population = Population([Tree(tree) for tree in self.initializer(**self.pi_init)])

        # Evaluate the initial population.
        population.calculate_semantics(inputs=X_train, testing=False)
        population.evaluate(target=y_train, testing=False)
        end = time.time()

        # Set up the elites
        self.elites, self.elite = self.find_elit_func(population, n_elites)   
        self.population = population

        # setting up log paths and run info
        self.log_level = log
        self.log_path = log_path
        self.run_info = run_info
        self.dataset = curr_dataset
        self.time_dict = {'mutation':[], 'xo':[]}

        # setting timeouts and iterations
        self.timeout = timeout
        self.iteration = 0

        # calculating the testing semantics and the elite's testing fitness if test_elite is true
        if test_elite:
            self.elite.evaluate(
                ffunction, X=X_test, y=y_test, testing=True
            )

        # Display and log results
        self.print_results(0, start, end) if verbose > 0 else None
        self.log_results(0, start, end)

        # Run callbacks
        for callback in self.callbacks:
            callback.on_train_start(self)

        start_time = time.time()

        # Main evolutionary loop.
        for it in range(1, n_iter + 1, 1):
            self.time_dict = {'mutation':[], 'xo':[]}
            self.iteration += 1

            if time.time() - start_time > self.timeout:
                print(f"Timeout reached at iteration {it}. Training stopped.") if verbose > 0 else None
                break
            
            # starting an empty offspring population
            offs_pop, start = [], time.time()

            # adding the elite to the offspring population, if applicable
            if elitism:
                offs_pop.extend(self.elites)

            # Run callbacks
            for callback in self.callbacks:
                callback.on_generation_start(self, it)

            while len(offs_pop) < self.pop_size:
                # XO selected
                if random.random() < self.p_xo:
                    offs = self.crossover_step()
                    offs_pop.extend(offs)

                # Mutation selected
                else:
                    offs = self.mutation_step()
                    offs_pop.append(offs)

            # Check if the offspring population is larger than the population size
            if len(offs_pop) > population.size:
                offs_pop = offs_pop[: population.size]

            # Evaluate the offspring population
            offs_pop = Population(offs_pop) 
            offs_pop.calculate_semantics(inputs=X_train, testing=False)
            offs_pop.evaluate(target=y_train, testing=False)

            # Replace the current population with the offspring population P = P'
            population = offs_pop
            self.population = population
            end = time.time()

            # Set the new elite
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            # Evaluate the elite on the testing data
            if test_elite:
                self.elite.evaluate(
                    ffunction, X=X_test, y=y_test, testing=True
                )

            # Display and log results
            self.print_results(it, start, end) if verbose > 0 else None
            self.log_results(it, start, end)

            # Run callbacks
            for callback in self.callbacks:
                callback.on_generation_end(self, it)

            if self.stop_training:
                print(f"{it} iterations completed. Training stopped by callback.") if verbose > 0 else None
                break

        # Run callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

    # ------------------------------------ Helper functions ------------------------------------
    def crossover_step(self):  
        start = time.time()
        while True:
            parent1, parent2 = self.selector(self.population), self.selector(self.population)
            if parent1 != parent2:
                break  
                
        offs = self.xo_operator(parent1, parent2)
        self.time_dict['xo'].append(time.time() - start)
        # print(parent1.depth, parent2.depth, offs[0].depth, offs[1].depth)
        return offs

    def mutation_step(self): 
        start = time.time()
        parent = self.selector(self.population)
        offs = self.mutator(parent)
        self.time_dict['mutation'].append(time.time() - start)
        return offs
    
    def print_results(self, iteration, start, end):
        params = {
            "dataset": self.dataset,
            "it": iteration,
            "train": self.elite.fitness,
            "test": self.elite.test_fitness,
            "time": end - start,
            "nodes": self.elite.nodes_count,
            "total_nodes": self.elite.total_nodes,
            "average_depth": np.round(np.mean([ind.depth for ind in self.population.population]), 2),
            "div": int(self.calculate_diversity()),
            "mut": f"{np.round(1000*np.mean([self.time_dict['mutation']]),2) if self.time_dict['mutation'] != [] else 'N/A'} ({len(self.time_dict['mutation'])})",
            "xo": f"{np.round(1000*np.mean([self.time_dict['xo']]),2) if self.time_dict['xo'] != [] else 'N/A'} ({len(self.time_dict['xo'])})",
        }
        
        verbose_reporter(
                params, 
                first =iteration == 0,
                precision=3, 
                col_width=14
        )

    def log_results(self, 
                    iteration, 
                    start_time, 
                    end_time):
        
        if self.log_level == 0:
            return
                
        if self.log_level in [2, 4]:
            gen_diversity = self.calculate_diversity(iteration)

        if self.log_level == 2:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                float(gen_diversity),
                self.log_level,
            ]
        elif self.log_level == 3:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                " ".join([str(ind.nodes_count) for ind in self.population.population]),
                " ".join([str(f) for f in self.population.fit]),
                self.log_level,
            ]
        elif self.log_level == 4:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                float(gen_diversity),
                np.std(self.population.fit),
                " ".join([str(ind.nodes_count) for ind in self.population.population]),
                " ".join([str(f) for f in self.population.fit]),
                self.log_level,
            ]
        else:
            add_info = [self.elite.test_fitness, self.elite.nodes_count, self.log_level]

        logger(
            self.log_path,
            iteration,
            self.elite.fitness,
            end_time - start_time,
            float(self.population.nodes_count),
            additional_infos=add_info,
            run_info=self.run_info,
            seed=self.seed,
        )

    def calculate_diversity(self):
        return gsgp_pop_div_from_vectors(self.population.train_semantics)
