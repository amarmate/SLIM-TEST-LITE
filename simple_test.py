from functions.experiments.MULTIGP.run_multi import run_multi
from functions.experiments.GP.run_gp import run_gp

import threading

class Args:
    def __init__(self, workers):
        self.workers = workers
        self.cs = None 
        self.ci = None 

def main():
    args_gp = Args(workers=4)
    args_multi = Args(workers=4) 

    thread_gp = threading.Thread(target=run_gp, args=(args_gp,))
    thread_multi = threading.Thread(target=run_multi, args=(args_multi,))

    thread_gp.start()
    thread_multi.start()

    thread_gp.join()
    thread_multi.join()

if __name__ == "__main__":
    main()
