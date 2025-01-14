from slim_gsgp_lib_np.main_slim import slim
# from slim_gsgp_lib_np.main_gsgp import gsgp 
# from slim_gsgp_lib_np.main_gp import gp
from slim_gsgp_lib_np.utils.utils import train_test_split
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.operators.simplifiers import simplify_individual
from slim_gsgp_lib_np.evaluators.fitness_functions import rmse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import os
from tqdm import tqdm
from functions.test_funcs import mape, nrmse, r_squared, mae, standardized_rmse


# ----------------------------------- SLIM ----------------------------------- #
def test_slim(X_train, y_train, X_test, y_test,
            args_dict=None,
            dataset_name='dataset_1', 
            n_elites=1,
            initializer='rhh',
            n_samples=30,
            scale=True,
            algorithm="SLIM+SIG1",
            verbose=0,
            show_progress=True,
            log = False, 
            timeout = 100,
            callbacks = None,
            simplify_threshold = None,
):    
    """

    Arguments
    ---------
    X_train: torch.tensor
        The input training data.
    y_train: torch.tensor
        The target training data.
    X_test: torch.tensor
        The input test data.
    y_test: torch.tensor
        The target test data.
    args_dict: dict
        A dictionary containing the hyperparameters for the SLIM algorithm.
    dataset_name: str
        The name of the dataset.
    n_samples: int     
        The number of samples to generate.
    n_elites: int
        The number of elites.
    initializer: str
        The initializer to use.
    iterations: int
        The number of iterations to perform.
    scale: bool
        Whether to scale the data or not.
    algorithm: str
        The SLIM algorithm to use.
    verbose: int
        The verbosity level.
    show_progress: bool
        Whether to show the progress bar or not.
    log: bool
        Whether to log the results or not.
    timeout: int
        The maximum time to train the model.
    callbacks: list
        A list containing the callbacks to use.
    simplify_threshold: float
        The threshold to use for simplification. 

    Returns
    -------
    rmse: list
        A list containing the RMSE scores.
    mape: list
        A list containing the MAPE scores.
    mae: list
        A list containing the MAE scores.
    rmse_compare: list
        A list containing the RMSE scores.
    mape_compare: list
        A list containing the MAPE scores.
    mae_compare: list
        A list containing the MAE scores.
    time_stats: list
        A list containing the time taken to train the model.
    size: list
        A list containing the size of the trees.
    representations: list
        A list containing the tree representations.
    """    
    rmse_, mae_, mape_, rmse_comp, mae_comp, mape_comp, time_stats, size, representations = [], [], [], [], [], [], [], [], []

    if scale:
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    for it in tqdm(range(n_samples), disable=not show_progress):
        if log:
            algorithm_name = 'MUL-' + algorithm.split('*')[1] if '*' in algorithm else 'ADD-' + algorithm.split('+')[1]
            path = f"logs/{dataset_name}/{algorithm_name}_{it}.log"
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            
        start = time.time()        
        try:
            final_tree = slim(X_train=X_train, y_train=y_train,
                                dataset_name=dataset_name, slim_version=algorithm, seed=it,
                                reconstruct=True, n_jobs=1, test_elite=False,
                                n_elites=n_elites, **args_dict, log_level=(3 if log else 0), log_path=(path if log else None),
                                timeout=timeout, callbacks=callbacks, verbose=0,
            )
        except Exception as e:
            print('Error during testing SLIM:', e)
            print('Args:', args_dict)
            continue

        end = time.time()   
        if simplify_threshold is not None:
            final_tree = simplify_individual(final_tree, y_train=y_train, X_train=X_train, threshold=simplify_threshold)
            
        # Get the node count of the tree
        nodes_count = final_tree.nodes_count
        time_taken = end - start
        
        # Calculate predictions and metrics
        y_pred = final_tree.predict(X_test)
        rmse_score = rmse(y_test, y_pred)
        mae_score = mae(y_test, y_pred)
        mape_score = mape(y_test, y_pred)
        rmse_compare = rmse_score
        mae_compare = mae_score
        mape_compare = mape_score
        
        if scale:
            # Apply inverse scaling
            y_pred_descaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
            y_test_descaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

            # Calculate metrics
            rmse_compare = rmse(y_test_descaled, y_pred_descaled)
            mae_compare = mae(y_test_descaled, y_pred_descaled)
            mape_compare = mape(y_test_descaled, y_pred_descaled)
                
        # Append metrics to respective lists
        rmse_.append(rmse_score)
        mape_.append(mape_score)
        mae_.append(mae_score)
        rmse_comp.append(rmse_compare)
        mape_comp.append(mape_compare)
        mae_comp.append(mae_compare)
        time_stats.append(time_taken)
        size.append(nodes_count)
        representations.append(final_tree.get_tree_representation())

        if verbose:
            print(f"Iteration {it+1}/{n_samples} - RMSE: {rmse_score:.4f} - SIZE: {nodes_count:.1f} - MAE: {mae_score:.4f} - Time: {time_taken:.2f}s")
        
    return rmse_, mape_, mae_, rmse_comp, mape_comp, mae_comp, time_stats, size, representations


# ----------------------------------- GSGP ----------------------------------- #
# def test_gsgp(X, y, args_dict=None,
#               dataset_name='dataset_1', 
#               iterations=30,
#               scale=True,
#               verbose=0,
#               p_train=0.7,
#               threshold=100000,
#               show_progress=True
# ):    
#     """
#     Arguments
#     ---------
#     X: torch.tensor
#         The input data.
#     y: torch.tensor
#         The target data.
#     args_dict: dict
#         A dictionary containing the hyperparameters for the GSGP algorithm.
#     dataset_name: str
#         The name of the dataset.
#     iterations: int
#         The number of iterations to perform.
#     scale: bool
#         Whether to scale the data or not.
#     verbose: int
#         The verbosity level.
#     p_train: float
#         The percentage of the training set.
#     threshold: int
#         The maximum number of nodes allowed in the tree.
#     show_progress: bool
#         Whether to show the progress bar or not.


#     Returns
#     -------
#     rmse: list
#         A list containing the RMSE scores.
#     mape: list
#         A list containing the MAPE scores.
#     nrmse: list
#         A list containing the NRMSE scores.
#     r2: list   
#         A list containing the R-squared scores.
#     mae: list  
#         A list containing the MAE scores.
#     std_rmse: list 
#         A list containing the standardized RMSE scores.
#     time_stats: list
#         A list containing the time taken to train the model.
#     train_fit: list
#         A list containing the training fitness scores.
#     test_fit: list
#         A list containing the test fitness scores.
#     size: list
#         A list containing the size of the trees.
#     """
    
#     rmse_, mape_, nrmse_, r2_, mae_, std_rmse_, time_stats, train_fit, test_fit, size = [], [], [], [], [], [], [], [], [], []

#     for it in tqdm(range(iterations), disable=not show_progress):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-p_train, seed=it)

#         if scale:
#             scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
#             X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
#             X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
#             y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
#             y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)

#         path = f"logs/{dataset_name}/GSGP_{it}.log"
#         if not os.path.exists(os.path.dirname(path)):
#             os.makedirs(os.path.dirname(path))
            
#         start = time.time()
#         final_tree = gsgp(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                           dataset_name=dataset_name, seed=it,
#                           log_path=path, verbose=verbose, **args_dict, reconstruct=True, log_level=0)
#         end = time.time()

#         if final_tree.nodes > threshold:
#             print(f"Tree too large: {final_tree.nodes}")
#             continue
        
#         # Get the node count of the tree
#         nodes_count = final_tree.nodes
#         time_taken = end - start
#         train_fitness_elite = final_tree.fitness.item()
#         test_fitness_elite = final_tree.test_fitness.item()

#         # Calculate predictions and metrics
#         y_pred = final_tree.predict(X_test)
            
#         rmse_score = rmse(y_test, y_pred).item()
#         mape_score = mape(y_test, y_pred)
#         nrmse_score = nrmse(y_test, y_pred)
#         r2_score = r_squared(y_test, y_pred)
#         mae_score = mae(y_test, y_pred)
#         std_rmse_score = standardized_rmse(y_test, y_pred)
        
#         # Append metrics to respective lists
#         rmse_.append(rmse_score)
#         mape_.append(mape_score)
#         nrmse_.append(nrmse_score)
#         r2_.append(r2_score)
#         mae_.append(mae_score)
#         std_rmse_.append(std_rmse_score)
#         time_stats.append(time_taken)
#         train_fit.append(train_fitness_elite)
#         test_fit.append(test_fitness_elite)
#         size.append(nodes_count)

#     return rmse_, mape_, nrmse_, r2_, mae_, std_rmse_, time_stats, train_fit, test_fit, size




# # ----------------------------------- GP ----------------------------------- #
# def test_gp(X, y, args_dict=None,
#             dataset_name='dataset_1',
#             iterations=30,
#             scale=True,
#             verbose=0,
#             p_train=0.7,
#             show_progress=True
# ):
    
#     """
#     Arguments
#     ---------
#     X: torch.tensor
#         The input data. 
#     y: torch.tensor
#         The target data.
#     args_dict: dict
#         A dictionary containing the hyperparameters for the GP algorithm.
#     dataset_name: str
#         The name of the dataset.
#     iterations: int
#         The number of iterations to perform.
#     scale: bool
#         Whether to scale the data or not.
#     verbose: int
#         The verbosity level.
#     p_train: float
#         The percentage of the training set.
#     show_progress: bool
#         Whether to show the progress bar or not.

#     Returns
#     -------

#     rmse: list
#         A list containing the RMSE scores.
#     mape: list
#         A list containing the MAPE scores.
#     nrmse: list
#         A list containing the NRMSE scores.
#     r2: list
#         A list containing the R-squared scores.
#     mae: list
#         A list containing the MAE scores.
#     std_rmse: list
#         A list containing the standardized RMSE scores.
#     time_stats: list
#         A list containing the time taken to train the model.
#     train_fit: list
#         A list containing the training fitness scores.
#     test_fit: list
#         A list containing the test fitness scores.
#     size: list
#         A list containing the size of the trees.
#     """

#     rmse_, mape_, nrmse_, r2_, mae_, std_rmse_, time_stats, train_fit, test_fit, size = [], [], [], [], [], [], [], [], [], []

#     for it in tqdm(range(iterations), disable=not show_progress):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-p_train, seed=it)

#         if scale:
#             scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
#             X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
#             X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
#             y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
#             y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)

#         path = f"logs/{dataset_name}/GP_{it}.log"
#         if not os.path.exists(os.path.dirname(path)):
#             os.makedirs(os.path.dirname(path))
            
#         start = time.time()
#         final_tree = gp(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                         dataset_name=dataset_name, seed=it,
#                         log_path=path, verbose=verbose, **args_dict, log_level=0)
#         end = time.time()

#         # Get the node count of the tree
#         nodes_count = final_tree.node_count
#         time_taken = end - start
#         train_fitness_elite = final_tree.fitness.item()
#         test_fitness_elite = final_tree.test_fitness.item()

#         # Calculate predictions and metrics
#         y_pred = final_tree.predict(X_test)
            
#         rmse_score = rmse(y_test, y_pred).item()
#         mape_score = mape(y_test, y_pred)
#         nrmse_score = nrmse(y_test, y_pred)
#         r2_score = r_squared(y_test, y_pred)
#         mae_score = mae(y_test, y_pred)
#         std_rmse_score = standardized_rmse(y_test, y_pred)
        
#         # Append metrics to respective lists
#         rmse_.append(rmse_score)
#         mape_.append(mape_score)
#         nrmse_.append(nrmse_score)
#         r2_.append(r2_score)

#         mae_.append(mae_score)
#         std_rmse_.append(std_rmse_score)
#         time_stats.append(time_taken)
#         train_fit.append(train_fitness_elite)
#         test_fit.append(test_fitness_elite)
#         size.append(nodes_count)

#     return rmse_, mape_, nrmse_, r2_, mae_, std_rmse_, time_stats, train_fit, test_fit, size