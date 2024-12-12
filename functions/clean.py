
import pickle
import os 

def clean():
    dataset_names=[]
    for file in os.listdir('results/slim'):
        if 'scaled_xo_mutxo_strucmut' in file and file.split('_scaled')[0] not in dataset_names:
            dataset_name = file.split('_scaled')[0]
            dataset_names.append(dataset_name)

    for dataset in dataset_names:
        dict_params = {}
        dict_results = {}
        
        for suffix in ['MUL_ABS', 'MUL_SIG1', 'MUL_SIG2', 'SUM_ABS', 'SUM_SIG1', 'SUM_SIG2']:
            params = pickle.load(open(f'params/{dataset}_SLIM_{suffix}_scaled_strucmut_new.pkl', 'rb'))
            dict_params.update(params)
            results = pickle.load(open(f'results/slim/{dataset}_SLIM_{suffix}_scaled_strucmut_new.pkl', 'rb'))
            for k, v in results.items():
                if k not in dict_results:
                    dict_results[k] = {}
                dict_results[k].update(v)
                
            # Delete the loaded pickle files 
            os.remove(f'params/{dataset}_SLIM_{suffix}_scaled_strucmut_new.pkl')
            os.remove(f'results/slim/{dataset}_SLIM_{suffix}_scaled_strucmut_new.pkl')
        
        # Dump the results 
        pickle.dump(dict_params, open(f'params/{dataset}_scaled_strucmut_new.pkl', 'wb'))
        pickle.dump(dict_results, open(f'results/slim/{dataset}_scaled_strucmut_new.pkl', 'wb'))