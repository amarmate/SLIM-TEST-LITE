import json
import os 
from functions.experiments.GP.config_gp import *

def get_tasks(args): 
    # Get the tasks to be executed
    tasks = [
        (dataset, name, selector, split_id, f'{name}-{selector}-{split_id}')
        for selector in SELECTORS
        for name, dataset in datasets.items()
        for split_id in range(N_SPLITS)
    ]
    total_tasks = len(tasks)

    # Check if some tasks have already been completed 
    if os.path.join('..', DATA_DIR, 'file_track.json'):
        with open(os.path.join('..', DATA_DIR, 'file_track.json'), 'r') as f:
            task_list = json.load(f)
        try: 
            tasks = [task for task in tasks if task_list[task[4]] == 'not_done']
        except KeyError:
            print("Task tracking file doesn't contain all tasks, running all tasks and creating a new tracking file.")
            track_file = {task[4]: 'not_done' for task in tasks}
            with open(os.path.join('..', DATA_DIR, 'file_track.json'), 'w') as f:
                json.dump(track_file, f, indent=4)
    else:
        print("No previous task tracking found, running all tasks and creating a new tracking file.")
        track_file = {task[4]: 'not_done' for task in tasks}
        with open(os.path.join('..', DATA_DIR, 'file_track.json'), 'w') as f:
            json.dump(track_file, f, indent=4)

    remaining_tasks = len(tasks)
    print(f"Running {remaining_tasks} out of {total_tasks} tasks.")

    if remaining_tasks == 0:
        print("All tasks have already been completed.")
        return []
    return tasks