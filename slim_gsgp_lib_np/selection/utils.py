import random 
import math

def calculate_musigma_pp(pp: float,
                              n_cases: int = 500,
                              n_samples: int = 250
                              ):
    """
    Faster approximation using Python's random module:
    - For each sample, generate n_cases of I ~ N(0, pp) via random.gauss
    - Compute softmax weights
    - Zero out weights < threshold
    - Count non-zero weights
    Returns: (mean_count, std_count)
    """
    counts = []
    for _ in range(n_samples):
        I = [random.gauss(0, pp) for _ in range(n_cases)]
        max_I = max(I)
        exp_I = [math.exp(i - max_I) for i in I]
        total = sum(exp_I)
        weights = [e / total for e in exp_I]
        count = sum(1 for w in weights if w >= 0.01)
        counts.append(count)
    mean_count = sum(counts) / n_samples
    var = sum((c - mean_count) ** 2 for c in counts) / (n_samples-1)
    std_count = math.sqrt(var)
    return mean_count, std_count

