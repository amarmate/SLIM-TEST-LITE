from slim_gsgp_lib.datasets.data_loader import *
from slim_gsgp_lib.utils.utils import train_test_split
from slim_gsgp_lib.main_slim import slim

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

if __name__ == '__main__':
    seed = 0
    datasets = [globals()[i] for i in globals() if 'load' in i][2:]
    X,y = datasets[2]()
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed=seed)

    example_tree = slim(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dataset_name='test',
                    slim_version='SLIM+SIG2',
                    max_depth=22, init_depth=10, pop_size=100, n_iter=50, seed=seed,
                    p_inflate=0.35, p_struct=0.35, test_elite=True, # selector='lexicase',
                    struct_mutation=True, decay_rate=0.4, p_xo=0, verbose=1)    