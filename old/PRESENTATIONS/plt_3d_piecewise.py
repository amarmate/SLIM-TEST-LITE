import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# --- Matplotlib Configuration for LaTeX Rendering ---
# IMPORTANT: You must have a LaTeX distribution (MiKTeX, TeX Live, MacTeX)
# installed on your system and in your PATH for this to work.
# Ghostscript and dvipng are also typically required.
try:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}", # For 'cases' environment and \text
        # Optional: Use LaTeX default fonts for consistency
        # "font.family": "serif",
        # "font.serif": ["Computer Modern Roman"], # Or another LaTeX font
    })
    print("Successfully configured Matplotlib to use LaTeX for text rendering.")
except Exception as e:
    print(f"Could not configure Matplotlib for LaTeX rendering: {e}")
    print("Ensure a LaTeX distribution is installed and in your PATH.")
    print("Falling back to Matplotlib's internal mathtext engine.")
    plt.rcParams["text.usetex"] = False


def plot_regions_2d(x_values, y_values, center_x, center_y, radius):
    """
    This function creates a 2D plot of the regions defined by the piecewise condition
    and displays the LaTeX definition of the function.
    The plot limits are set from -1 to 2 for both x and y axes.
    """
    X, Y = np.meshgrid(x_values, y_values)

    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    condition1_mask = distance < radius

    # Changed figsize to be square for better use with axis('equal') and fixed limits
    fig, ax = plt.subplots(figsize=(8, 8))

    region_map = np.zeros(X.shape)
    region_map[condition1_mask] = 1

    ax.imshow(region_map, extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
               origin='lower', cmap='coolwarm', alpha=0.7)

    circle_boundary = plt.Circle((center_x, center_y), radius, color='black', fill=False, linestyle='--', linewidth=1.5)
    ax.add_artist(circle_boundary)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('2D Plot of Regions and Function Definition')

    # Set x and y limits as requested
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])

    # Apply 'equal' aspect ratio after setting limits
    ax.set_aspect('equal', adjustable='box') # 'box' will change axis limits to make data aspect equal

    # LaTeX for legend labels
    legend_label1 = rf'$| |(x,y)-({center_x:.2f},{center_y:.2f})| | < {radius:.2f}$' # Using f-string for numbers
    legend_label2 = rf'$| |(x,y)-({center_x:.2f},{center_y:.2f})| | \geq {radius:.2f}$'

    region1_patch = mpatches.Patch(color=plt.cm.coolwarm(1.0), label=legend_label1)
    region2_patch = mpatches.Patch(color=plt.cm.coolwarm(0.0), label=legend_label2)
    ax.legend(handles=[region1_patch, region2_patch], loc='upper right', fontsize='small')

    ax.grid(True, linestyle=':', alpha=0.5)
    # ax.axis('equal') # Replaced by ax.set_aspect('equal', adjustable='box') after limits

    func_f1 = r"3x^2 - y + 5"
    func_f2 = r"2x - y"
    condition_text = rf"\text{{if }} \sqrt{{(x-{center_x:.2f})^2 + (y-{center_y:.2f})^2}} < {radius:.2f}"
    otherwise_text = r"\text{otherwise}"

    if plt.rcParams['text.usetex']:
        latex_string = (
            rf"$f(x,y) = \begin{{cases}} "
            rf"{func_f1} & {condition_text} \\ "
            rf"{func_f2} & {otherwise_text} "
            rf"\end{{cases}}$"
        )
    else:
        print("Warning: text.usetex is False. Displaying simplified function definition.")
        latex_string = (
            f"f(x,y):\n"
            f"{func_f1} if sqrt((x-{center_x:.2f})^2+(y-{center_y:.2f})^2) < {radius:.2f}\n"
            f"{func_f2} otherwise"
        )

    fig.text(0.5, 0.02, latex_string, ha='center', va='bottom', fontsize=12, color='black', # Increased vertical position a bit
             bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    # Adjust layout after all elements, including text, are added
    plt.tight_layout(rect=[0, 0.1, 1, 0.93]) # Adjusted rect for text and title
    plt.show()


def plot_piecewise_3d_and_regions():
    """
    This function creates a 3D plot of a piecewise function and
    a 2D plot of its regions including the LaTeX definition.
    """
    x_vals = np.linspace(-1, 2, 200) # These already match the desired limits
    y_vals = np.linspace(-1, 2, 200) # These already match the desired limits
    c_x = 0.5
    c_y = 0.3
    rad = 0.7

    plot_regions_2d(x_vals, y_vals, c_x, c_y, rad)

    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    Z_surface = np.zeros(X_mesh.shape)
    distance_mesh = np.sqrt((X_mesh - c_x)**2 + (Y_mesh - c_y)**2)

    condition1_surface = distance_mesh < rad
    Z_surface[condition1_surface] = 3 * X_mesh[condition1_surface]**2 - Y_mesh[condition1_surface] + 5
    condition2_surface = np.logical_not(condition1_surface)
    Z_surface[condition2_surface] = 2 * X_mesh[condition2_surface] - Y_mesh[condition2_surface]

    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot_surface(X_mesh, Y_mesh, Z_surface, cmap='viridis', edgecolor='none')
    ax_3d.set_xlabel('$x$')
    ax_3d.set_ylabel('$y$')
    ax_3d.set_zlabel('$f(x, y)$')
    ax_3d.set_title('3D Plot of the Piecewise Function')
    plt.show()

if __name__ == '__main__':
    plot_piecewise_3d_and_regions()