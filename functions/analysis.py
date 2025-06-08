import pickle
import pandas as pd
import matplotlib.pyplot as plt
from functions.misc_functions import pf_rmse_comp_extended, pf_rmse_comp_time


# ---------------------------------------------------- PARETO FRONT PLOTTING ----------------------------------------------------
def plot_pareto_front(dataset_name, selector, suffix):
    """
    Loads a Pareto front from a pickle file, processes it, and generates a scatter plot
    with time annotations for each point.

    The function assumes the Pareto front data, after processing by
    pf_rmse_comp_extended and pf_rmse_comp_time, will be a list of tuples/lists,
    where each inner collection has at least three elements: RMSE, size, and time,
    in that order.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'airfoil').
        selector (str): The selector string (e.g., 'dalex_size_2').
        suffix (int or str): The suffix for the pickle file (e.g., 1).
    """
    file_path = f'experiment_results/{dataset_name}/{selector}_pf{suffix}.pkl'
    plot_title = f'Pareto Front for {dataset_name.capitalize()} Dataset ({selector})'
    xlabel = 'Size'
    ylabel = 'RMSE'
    colorbar_label = 'Time (s)' # Assuming 'time' is in seconds, adjust if needed

    print(f"Loading data from: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            pareto_data = pickle.load(f)

        processed_pareto = pf_rmse_comp_extended(pareto_data)
        final_pareto_list = pf_rmse_comp_time(processed_pareto)

        pareto_df = pd.DataFrame(final_pareto_list, columns=['rmse', 'size', 'time'])

        if pareto_df.empty:
            print(f"No data to plot for {file_path} after processing.")
            return

        # Generate the plot
        plt.figure(figsize=(12, 7)) # Slightly larger figure for annotations
        scatter = plt.scatter(
            pareto_df['size'],
            pareto_df['rmse'],
            c=pareto_df['time'],
            cmap='viridis',
            s=80,
            edgecolor='k',
            alpha=0.7 # Added alpha for better visibility if annotations overlap points
        )
        plt.colorbar(scatter, label=colorbar_label)
        plt.title(plot_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)

        # Add annotations for each point
        for i, row in pareto_df.iterrows():
            plt.annotate(
                f"{row['time']:.0f}s",  
                (row['size'], row['rmse']), 
                textcoords="offset points", 
                xytext=(5,5),  
                ha='left',  
                fontsize=6,  
                color='black'
            )

        plt.tight_layout() # Adjust layout to make room for annotations
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# ---------------------------------------------------