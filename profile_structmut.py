from slim_gsgp_lib.datasets.data_loader import *
from sklearn.preprocessing import MinMaxScaler
from slim_gsgp_lib.utils.utils import train_test_split
from slim_gsgp_lib.main_slim import slim
import cProfile
import pstats


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    
    seed = 0
    datasets = [globals()[i] for i in globals() if 'load' in i][2:]
    X,y = datasets[2]()
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X = torch.tensor(scaler_X.fit_transform(X))
    y = torch.tensor(scaler_y.fit_transform(y.reshape(-1,1)).reshape(-1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed=seed)

    example_tree = slim(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dataset_name='test',
                    max_depth=22, init_depth=10, pop_size=200, n_iter=10, seed=seed,
                    p_inflate=0.6, p_struct=0.3, test_elite=True, selector='lexicase',
                    struct_mutation=True, decay_rate=0.4, p_xo=0, type_structure_mutation='new', verbose=1)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('time').print_stats(20)
    stats.dump_stats('profile_structmut')
    