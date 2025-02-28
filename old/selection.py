import timeit
import random
import numpy as np

# Dummy classes to simulate the Population and Individual structures
class DummyIndividual:
    def __init__(self, id, num_cases):
        # Simulate error values for each test case
        self.errors_case = np.random.rand(num_cases)

class DummyPopulation:
    def __init__(self, num_individuals, num_cases):
        # Create a list of dummy individuals
        self.population = [DummyIndividual(i, num_cases) for i in range(num_individuals)]
        # Build a 2D array of errors (individuals x test cases)
        self.errors_case = np.array([ind.errors_case for ind in self.population])
        # Simulate fitness values
        self.fit = np.random.rand(num_individuals)

# Example targets array (assume one target per test case)
num_cases = 5000
targets = np.random.rand(num_cases)

# Dummy population parameters
num_individuals = 500
dummy_pop = DummyPopulation(num_individuals, num_cases)



def old_epsilon_lexicase_selection(targets, eps_fraction=1e-7, mode='min'):
    # Helper: Compute the epsilon threshold using fitness values.
    def compute_epsilon(fitness_values, eps_fraction):
        return eps_fraction * np.std(fitness_values)

    # Helper: Randomly select a test case index.
    def select_random_case(num_cases):
        return random.randint(0, num_cases - 1)

    # Helper: Compute squared errors for a given test case.
    def compute_case_errors(errors, case_idx):
        return errors[:, case_idx] ** 2

    # Helper: Filter candidates based on the epsilon threshold and mode.
    def filter_candidates(case_errors, epsilon, mode):
        if mode == 'min':
            best_value = np.min(case_errors)
            indices = np.where(case_errors <= best_value + epsilon)[0]
        elif mode == 'max':
            best_value = np.max(case_errors)
            indices = np.where(case_errors >= best_value - epsilon)[0]
        else:
            raise ValueError("Invalid mode. Use 'min' or 'max'.")
        return indices

    # Helper: Update the pool and errors array based on filtered indices.
    def update_pool_and_errors(pool, errors, indices):
        new_pool = [pool[i] for i in indices]
        new_errors = np.array([ind.errors_case for ind in new_pool])
        return new_pool, new_errors

    # Helper: Finalize selection when multiple candidates remain.
    def finalize_selection(pool):
        if len(pool) == 1:
            return pool[0]
        return random.choice(pool)

    @profile
    def els(pop):
        # Retrieve the initial errors array and compute epsilon.
        errors = pop.errors_case
        epsilon = compute_epsilon(pop.fit, eps_fraction)
        num_cases = targets.shape[0]
        
        # Start with a full copy of the population.
        pool = pop.population.copy()
        
        # Iterate for a fixed number of cases (here, 5 iterations).
        for _ in range(5):
            case_idx = select_random_case(num_cases)
            case_errors = compute_case_errors(errors, case_idx)
            indices = filter_candidates(case_errors, epsilon, mode)
            
            # If a single candidate remains, return it immediately.
            if indices.size == 1:
                return pool[indices[0]]
            
            # Update the pool and corresponding errors array.
            pool, errors = update_pool_and_errors(pool, errors, indices)
        
        return finalize_selection(pool)
    
    return els



def new_epsilon_lexicase_selection(targets, eps_fraction=1e-7, mode='min'):
    # Helper: Compute the epsilon threshold using fitness values.
    def compute_epsilon(fitness_values, eps_fraction):
        return eps_fraction * np.std(fitness_values)

    # Helper: Randomly select a test case index.
    def select_random_case(num_cases):
        return random.randint(0, num_cases - 1)

    # Helper: Compute squared errors for the current candidate indices on a given test case.
    def compute_case_errors(errors, candidate_indices, case_idx):
        return errors[candidate_indices, case_idx] ** 2

    # Helper: Filter candidate indices based on the epsilon threshold and mode.
    def filter_candidates(current_errors, epsilon, mode):
        if mode == 'min':
            best_value = np.min(current_errors)
            mask = current_errors <= best_value + epsilon
        elif mode == 'max':
            best_value = np.max(current_errors)
            mask = current_errors >= best_value - epsilon
        else:
            raise ValueError("Invalid mode. Use 'min' or 'max'.")
        return mask

    # Helper: Finalize selection based on the remaining candidate indices.
    def finalize_selection(candidate_indices, population):
        if candidate_indices.size == 0:
            # Fallback: if no candidate remains, choose one at random.
            return random.choice(population)
        if candidate_indices.size == 1:
            return population[candidate_indices[0]]
        return population[random.choice(candidate_indices.tolist())]

    @profile
    def els(pop):
        errors = pop.errors_case    # NumPy array of shape (N, num_cases)
        epsilon = compute_epsilon(pop.fit, eps_fraction)
        num_cases = targets.shape[0]
        candidate_indices = np.arange(len(pop.population))
        
        for _ in range(5):
            case_idx = select_random_case(num_cases)
            current_errors = compute_case_errors(errors, candidate_indices, case_idx)
            mask = filter_candidates(current_errors, epsilon, mode)
            candidate_indices = candidate_indices[mask]
            if candidate_indices.size <= 1:
                break
        
        return finalize_selection(candidate_indices, pop.population)
    
    return els


if __name__ == '__main__':
    old_selector = old_epsilon_lexicase_selection(targets, eps_fraction=1e-7, mode='min')
    new_selector = new_epsilon_lexicase_selection(targets, eps_fraction=1e-7, mode='min')

    # Define the number of repetitions for the benchmark
    repetitions = 10000

    # Benchmark the old implementation
    print("Old implementation:")
    old_time = timeit.timeit(lambda: old_selector(dummy_pop), number=repetitions)
    print(f"Total time: {old_time:.6f} seconds")

    # Benchmark the new implementation
    print("\nNew implementation:")
    new_time = timeit.timeit(lambda: new_selector(dummy_pop), number=repetitions)
    print(f"Total time: {new_time:.6f} seconds")


