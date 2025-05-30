import os 
import argparse

def parse_args(config): 
    """
    Parse command line arguments for running GP experiments.
        
    Args:
        config (dict): Configuration dictionary containing experiment settings. 
    Returns:
        argparse.Namespace: Parsed command line arguments.
    Raises:
        ValueError: If the number of workers exceeds available CPU cores or if chunk size and index are not provided correctly.
        argparse.ArgumentError: If chunk size is less than the number of splits.
    Arguments:
        --workers (int): Number of parallel workers to use. Defaults to 0 (all available).
        --cs (int): Number of experiments per chunk. Must be provided with --ci.
        --ci (int): Zero-based index of the chunk to run. Must be provided with --cs.
    Usage example:
        python run_gp.py --workers 4 --cs 10 --ci 0
    """
    parser = argparse.ArgumentParser(description='Run GP experiments')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (default: all available)')
    parser.add_argument('--cs', type=int, default=None,
                        help='Number of experiments per chunk')
    parser.add_argument('--ci', type=int, default=None,
                        help='Zero-based index of the chunk to run')

    args = parser.parse_args()
    if args.workers > os.cpu_count():
        raise ValueError(f"workers cannot be greater than available CPU cores ({os.cpu_count()})")
    if (args.cs is None) ^ (args.ci is None):
        parser.error("Both --cs (chunk_size) and --ci (chunk_idx) must be provided together")
    if args.cs is not None and args.cs < config['N_SPLITS']:
        parser.error(f"--cs (chunk_size) must be at least N_SPLITS ({config['N_SPLITS']})")

    # Add default values for chunk size and index if not provided
    if args.workers is None or args.workers <= 0:
        args.workers = int(os.cpu_count() * 0.9)

    return args