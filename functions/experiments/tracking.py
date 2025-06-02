import os 
from pathlib import Path
from slim_gsgp_lib_np.utils.utils import train_test_split

def get_tasks(args, config): 
    """
    Get the tasks to be executed based on the provided configuration and command line arguments.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
        config (dict): Configuration dictionary containing experiment settings.
    
    Returns:
        list: A list of tasks to be executed, where each task is a dictionnary containing:
            - gen_params: A dictionary with parameters for the genetic algorithm.
            - split_id: The split identifier for the task.
            - mask: The mask for the task, if applicable.
    """


    data_dir = Path("..") / config['DATA_DIR'] / config['EXPERIMENT_NAME']
    if not data_dir.exists():
        os.mkdir(data_dir)
        print(f"Created directory: {data_dir}")
        
    previous_dir = os.getcwd()
    os.chdir(data_dir)

    tasks = [
        (dataset, name, selector, split_id, f'{name}-{selector}-{split_id}')
        for selector in config['SELECTORS']
        for name, dataset in config['datasets'].items()
        for split_id in range(config['N_SPLITS'])
    ]
    total_tasks = len(tasks)

    if args.cs is not None:
        cs, id = args.cs, args.ci
        tasks = tasks[cs * id: cs * (id + 1)]
        print(f"Running chunk {id} with {len(tasks)} tasks out of {total_tasks}")

    for task in tasks: 
        _, dataset_name, selector, split_id, task_name = task

        directory = Path(config['TEST_DIR']) / dataset_name / selector / f"checkpoint_testing_split{split_id}_{config['SUFFIX_SAVE']}.parquet"
        if directory.exists():
            print(f"Task {task_name} already completed. Skipping...")
            tasks.remove(task)
        else:
            print(f"Task {task_name} is pending.")
    
    os.chdir(previous_dir)

    print(f"Total tasks: {len(tasks)} out of {total_tasks}")
    tasks = [split_task(task, config) for task in tasks]

    return tasks 


def split_task(task, config): 
    """
    Split the dataset in a task for ease of use

    Uses the split_id as the random seed for reproducibility.
    Args:
        task (tuple): A tuple containing the dataset loader, name, selector, split_id, and task_name.
        config (dict): Configuration dictionary containing experiment settings.
    Returns:
        tuple: A tuple containing the training and testing datasets, along with the training mask if applicable.
        train, test, mask, name, selector, split_id
    """

    loader, name, selector, split_id, _ = task

    # Load the dataset
    dataset = loader()

    if len(dataset) == 2: 
        X_train, X_test, y_train, y_test = train_test_split(*dataset, p_test=config['P_TEST'], seed=split_id, indices_only=False)
        mask = None
    
    else: 
        idx_train, idx_test = train_test_split(dataset[0], dataset[1], p_test=config['P_TEST'], seed=split_id, indices_only=True)
        X_train, X_test = dataset[0][idx_train], dataset[0][idx_test]
        y_train, y_test = dataset[1][idx_train], dataset[1][idx_test]
        mask = [sbmask[idx_train] for sbmask in dataset[3]]

    update_dict = { 
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'dataset_name': name,
        'selector': selector,
    }
    
    gen_params = config['gen_params'].copy()
    gen_params.update(update_dict)

    return {
        'gen_params': gen_params,
        'split_id': split_id,
        'mask': mask
    }
