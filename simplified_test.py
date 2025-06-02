from functions.experiments.GP.run_gp import run_gp
from functions.experiments.GP.config_gp import config

from functions.experiments.parse import parse_args
from functions.experiments.github import init_or_update_repo

if __name__ == "__main__":
    args = parse_args(config)

    init_or_update_repo(config)
    
    run_gp(args)
