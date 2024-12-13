from matplotlib import pyplot as plt

class SLIM_GSGP_Callback:
    """
    Base class for callbacks.

    Methods
    -------
    on_train_start(slim_gsgp)
        Called at the beginning of the training process.
    on_train_end(slim_gsgp)
        Called at the end of the training process.
    on_generation_start(slim_gsgp, generation)
        Called at the beginning of each generation.
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    """

    def on_train_start(self, slim_gsgp):
        """
        Called at the beginning of the training process.
        """
        pass

    def on_train_end(self, slim_gsgp):
        """
        Called at the end of the training process.
        """
        pass

    def on_generation_start(self, slim_gsgp, iteration):
        """
        Called at the beginning of each generation.
        """
        pass

    def on_generation_end(self, slim_gsgp, iteration, start, end):
        """
        Called at the end of each generation.
        """
        pass


class LogDiversity(SLIM_GSGP_Callback):
    """
    Callback to log the diversity of the population.
    
    Attributes
    ----------
    diversity_structure : list
        List to store the diversity of the structure of the individuals.
    diversity_semantics : list
        List to store the diversity of the semantics of the individuals.

    Methods
    -------
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    plot_diversity()    
        Plot the diversity of the population.
    """

    def __init__(self):
        self.diversity_structure = []
        self.diversity_semantics = []

    def on_generation_end(self, slim_gsgp, generation, *args):
        individual_structure = [individual.structure[0] for individual in slim_gsgp.population.population]
        self.diversity_structure.append(len(set(individual_structure)))
        self.diversity_semantics.append(slim_gsgp.calculate_diversity(generation))

    def plot_diversity(self):
        fig, axs = plt.subplots(1,2, figsize=(18, 5))
        fig.suptitle('Diversity of the population')
        axs[0].plot(self.diversity_structure)   
        axs[0].set_title('Structure diversity')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Number of different structures')

        axs[1].plot(self.diversity_semantics)
        axs[1].set_title('Semantics diversity')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Diversity')
        plt.show()


class LogFitness(SLIM_GSGP_Callback):
    """
    Callback to log the fitness of the best individual in the population.

    Attributes
    ----------
    test_fitness : list
        List to store the test fitness of the best individual in the population.
    train_fitness : list
        List to store the train fitness of the best individual in the population.

    Methods
    -------
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    plot_fitness()
        Plot the fitness of the best individual in the population.
    """
    
    def __init__(self):
        self.test_fitness = []
        self.train_fitness = []

    def on_generation_end(self, slim_gsgp, generation, *args):
        self.test_fitness.append(slim_gsgp.elite.test_fitness)
        self.train_fitness.append(slim_gsgp.elite.fitness)

    def plot_fitness(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.test_fitness, label='Test fitness')
        plt.plot(self.train_fitness, label='Train fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()
        
class LogAge(SLIM_GSGP_Callback):
    """
    Callback to log the age of all the individuals in the population.

    Attributes
    ----------
    age : list
        List to store the age of all the individuals in the population in the last generation.

    Methods
    -------
    on_train_end(slim_gsgp)
        Called at the end of the training process.
    plot_age()
        Plot the age distribution of the population.    
    """
    def __init__(self):
        self.age = []

    def on_train_end(self, slim_gsgp, *args):
        self.age.append([individual.age for individual in slim_gsgp.population.population])

    def plot_age(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.age[-1], bins=20)
        plt.xlabel('Age')
        plt.ylabel('Number of individuals')
        plt.show()
        
        
        
def EarlyStopping(patience=10):
    """
    Callback to stop the training process when the fitness of the best individual does not improve for a number of generations.

    Attributes
    ----------
    patience : int
        Number of generations without improvement to wait before stopping the training process.

    Methods
    -------
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    """

    class EarlyStoppingCallback(SLIM_GSGP_Callback):
        def __init__(self):
            self.best_fitness = None
            self.counter = 0

        def on_generation_end(self, slim_gsgp, generation, *args):
            if generation == 1:
                # Reinicialize the counter and the best fitness
                self.best_fitness = slim_gsgp.elite.test_fitness.item()
                self.counter = 0
                
            elif self.best_fitness is None or slim_gsgp.elite.test_fitness.item() < self.best_fitness:
                self.best_fitness = slim_gsgp.elite.test_fitness.item()
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= patience:
                slim_gsgp.stop_training = True

    return EarlyStoppingCallback()


def EarlyStopping_train(patience=10):
    """
    Callback to stop the training process when the train fitness of the best individual in the training set does not improve for a number of generations.

    Attributes
    ----------
    patience : int
        Number of generations without improvement to wait before stopping the training process.

    Methods
    -------
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    """

    class EarlyStoppingCallback(SLIM_GSGP_Callback):
        def __init__(self):
            self.best_fitness = None
            self.counter = 0

        def on_generation_end(self, slim_gsgp, generation, *args):
            if generation == 1:
                # Reinicialize the counter and the best fitness
                self.best_fitness = slim_gsgp.elite.fitness.item()
                self.counter = 0
                
            elif self.best_fitness is None or slim_gsgp.elite.fitness.item() < self.best_fitness:
                self.best_fitness = slim_gsgp.elite.fitness.item()
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= patience:
                slim_gsgp.stop_training = True

    return EarlyStoppingCallback()
    