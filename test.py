from functions.test_algorithms import *
from functions.random_search import * 
from slim_gsgp_lib.datasets.data_loader import *
import pickle

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

pop_size = 100 
n_iter = 100
n_iter_rs = 10
n_iter_test = 3
p_train = 0.7

for dataset_loader in tqdm(datasets):
    X, y = dataset_loader()
    dataset_name = dataset_loader.__name__.split('load_')[1]
    
    # Perform random search for both scaled and unscaled versions
    print(f"Performing random search for {dataset_name}...")
    results_unscaled = random_search_slim(X, y, dataset_name, scale=False, p_train=p_train,
                                          iterations=n_iter_rs, pop_size=pop_size, n_iter=n_iter,
                                          struct_mutation=False, show_progress=False)
    
    results_scaled = random_search_slim(X, y, dataset_name, scale=True, p_train=p_train,
                                        iterations=n_iter_rs, pop_size=pop_size, n_iter=n_iter,
                                        struct_mutation=False, show_progress=False)
    
    print(f"Random search for {dataset_name} completed!")
    
    # Initialize dictionaries for scaled and unscaled results
    metrics = ['rmse_', 'mape_', 'nrmse_', 'r2_', 'mae_', 'std_rmse_', 'time_stats', 'train_fit', 'test_fit', 'size']
    results_scaled_dict = {metric: {} for metric in metrics}
    results_unscaled_dict = {metric: {} for metric in metrics}

    for algorithm in results_unscaled:
        # Retrieve the best hyperparameters for testing
        args_unscaled = results_unscaled[algorithm]
        args_scaled = results_scaled[algorithm]

        # Test SLIM for unscaled data
        rm_un, ma_un, nrmse_un, r2_un, mae_un, std_rmse_un, time_un, train_un, test_un, size_un = test_slim(
            X=X, y=y, args_dict=args_unscaled, dataset_name=dataset_loader.__name__,
            ms_lower=0, ms_upper=1, n_elites=1,
            iterations=n_iter_test, struct_mutation=False, scale=False, algorithm=algorithm,
            verbose=0, p_train=p_train, show_progress=False,
        )
        
        # Test SLIM for scaled data
        rm_sc, ma_sc, nrmse_sc, r2_sc, mae_sc, std_rmse_sc, time_sc, train_sc, test_sc, size_sc = test_slim(
            X=X, y=y, args_dict=args_scaled, dataset_name=dataset_loader.__name__,
            ms_lower=0, ms_upper=1, n_elites=1,
            iterations=n_iter_test, struct_mutation=False, scale=True, algorithm=algorithm,
            verbose=0, p_train=p_train, show_progress=False,
        )
        
        # Initialize storage for each algorithm if not already present
        for metric in metrics:
            if algorithm not in results_scaled_dict[metric]:
                results_scaled_dict[metric][algorithm] = []
                results_unscaled_dict[metric][algorithm] = []

        # Store scaled results
        results_scaled_dict['rmse_'][algorithm].extend(rm_sc)
        results_scaled_dict['mape_'][algorithm].extend(ma_sc)
        results_scaled_dict['nrmse_'][algorithm].extend(nrmse_sc)
        results_scaled_dict['r2_'][algorithm].extend(r2_sc)
        results_scaled_dict['mae_'][algorithm].extend(mae_sc)
        results_scaled_dict['std_rmse_'][algorithm].extend(std_rmse_sc)
        results_scaled_dict['time_stats'][algorithm].extend(time_sc)
        results_scaled_dict['train_fit'][algorithm].extend(train_sc)
        results_scaled_dict['test_fit'][algorithm].extend(test_sc)
        results_scaled_dict['size'][algorithm].extend(size_sc)

        # Store unscaled results
        results_unscaled_dict['rmse_'][algorithm].extend(rm_un)
        results_unscaled_dict['mape_'][algorithm].extend(ma_un)
        results_unscaled_dict['nrmse_'][algorithm].extend(nrmse_un)
        results_unscaled_dict['r2_'][algorithm].extend(r2_un)
        results_unscaled_dict['mae_'][algorithm].extend(mae_un)
        results_unscaled_dict['std_rmse_'][algorithm].extend(std_rmse_un)
        results_unscaled_dict['time_stats'][algorithm].extend(time_un)
        results_unscaled_dict['train_fit'][algorithm].extend(train_un)
        results_unscaled_dict['test_fit'][algorithm].extend(test_un)
        results_unscaled_dict['size'][algorithm].extend(size_un)

        print(f"Results for {algorithm} on {dataset_name} saved!")

    # Save the results to disk
    if not os.path.exists("output/SLIM"):
        os.makedirs("output/SLIM")

    with open(f"output/SLIM/{dataset_name}_scaled.pkl", 'wb') as f:
        pickle.dump(results_scaled_dict, f)

    with open(f"output/SLIM/{dataset_name}_unscaled.pkl", 'wb') as f:
        pickle.dump(results_unscaled_dict, f)

    print(f"Results for {dataset_name} saved!")
    print("---------------------------------------------------")

