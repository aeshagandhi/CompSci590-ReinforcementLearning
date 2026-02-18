import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython import display
import time
import json
from copy import deepcopy

plt.rc('font', size=30)  # controls default text sizes
plt.rc('axes', titlesize=25)  # fontsize of the axes title
plt.rc('axes', labelsize=25)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=17)  # fontsize of the tick labels
plt.rc('ytick', labelsize=17)  # fontsize of the tick labels
plt.rc('legend', fontsize=20)  # legend fontsize
plt.rc('figure', titlesize=30)
plt.tight_layout()

def plot_convergence(deltas, theta=None):
    plt.figure(figsize=(10, 6))
    
    # plot the convergence curve
    plt.plot(deltas, marker='o', markersize=4, linestyle='-', linewidth=1.5, label='Bellman Residual (δ)')
    
    # use log scale, because the error usually decreases exponentially, the log scale can see the convergence slope more clearly
    plt.yscale('log')
    
    # if theta is provided, draw a horizontal reference line
    if theta is not None:
        plt.axhline(y=theta, color='r', linestyle='--', alpha=0.7, label=f'Threshold (theta={theta})')
    
    # decorate the chart
    plt.title('Convergence of Value Iteration', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Max Absolute Difference (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    plt.show()


def plot(V, pi):
    # plot value
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.axis('on')
    ax1.cla()
    states = np.arange(V.shape[0])
    ax1.bar(states, V, edgecolor='none')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Value', rotation='horizontal', ha='right')
    ax1.set_title('Value Function')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax1.yaxis.grid()
    ax1.set_ylim(bottom=V.min())
    
    # plot policy
    ax2.axis('on')
    ax2.cla()
    plot_policy(pi, ax=ax2)
    # ax2.bar(states, pi, edgecolor='none')
    # ax2.set_xlabel('State')
    # ax2.set_ylabel('Action', rotation='horizontal', ha='right')
    # ax2.set_title('Policy')
    #
    # start, end = ax2.get_xlim()
    # ax2.xaxis.set_ticks(np.arange(start, end), minor=True)
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    # ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    # start, end = ax2.get_ylim()
    # ax2.yaxis.set_ticks(np.arange(start, end), minor=True)
    # ax2.grid(which='minor')

    divider = make_axes_locatable(ax2)
    
    fig.subplots_adjust(wspace=0.5)
    display.clear_output(wait=True)
    display.display(fig)
    time.sleep(0.001)
    plt.close()


def plot_gridworld_u(value_vector):
    # Create a grid full of NaNs
    grid = np.full((3, 4), np.nan)

    # Fill the grid with the values, skipping the obstacle position (1,1)
    grid[0, :] = value_vector[:4]  # First row
    grid[1, 0] = value_vector[4]  # Second row, first column
    grid[1, 2:] = value_vector[5:7]  # Second row, after obstacle
    grid[2, :] = value_vector[7:]  # Third row

    # Flip the grid on the y-axis
    grid = np.flipud(grid)

    fig, ax = plt.subplots()
    ax.matshow(grid, cmap='coolwarm')

    # Loop over data dimensions and create text annotations.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                # Get the color of the current cell
                cell_color = ax.matshow(grid, cmap='coolwarm_r').get_cmap()(grid[i, j])
                # Calculate the brightness of the color
                brightness = np.sqrt(0.299*cell_color[0]**2 + 0.587*cell_color[1]**2 + 0.114*cell_color[2]**2)
                text_color = 'w' if brightness < 0.5 else 'k'

                # Adjust the text position after flipping
                ax.text(j, i, f'{grid[i, j]:.2f}',
                        ha='center', va='center', color=text_color)
    # Set x and y labels
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels(np.arange(1, grid.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, grid.shape[0] + 1)[::-1])  # Reverse the order for y labels

    # Move x-axis labels to the bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    ax.set_title('Value Function')
    plt.show()

def plot_policy(policy_vector, ax=None):
    """
    Plots the policy for a gridworld with dividers between each state for improved visualization.

    Parameters:
    - grid_shape: tuple, the shape of the grid (rows, columns)
    - policy_vector: list, the deterministic policy actions for each state, indexed by the state number.
                     Actions are encoded as [0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT']
    """
    grid_shape = (3, 4)
    grid = np.full(grid_shape, np.nan)

    # Fill the grid with the values, skipping the obstacle position (1,1)
    grid[0, :] = policy_vector[:4]  # First row
    grid[1, 0] = policy_vector[4]  # Second row, first column
    grid[1, 2:] = policy_vector[5:7]  # Second row, after obstacle
    grid[2, :] = policy_vector[7:]  # Third row

    # Flip the grid on the y-axis
    # grid = np.flipud(grid)

    # Define the action arrows
    action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}  # Arrows representing each action

    # Create the figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the grid dividers
    for x in range(grid_shape[1] + 1):
        ax.axvline(x, color='black', linewidth=1)
    for y in range(grid_shape[0] + 1):
        ax.axhline(y, color='black', linewidth=1)

    # Set the range of the axes
    ax.set_xlim(0, grid_shape[1])
    ax.set_ylim(0, grid_shape[0])

    # Remove ticks and labels
    ax.set_xticks(np.arange(0, grid_shape[1] + 1, 1))
    ax.set_yticks(np.arange(0, grid_shape[0] + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Policy')

    # Create the policy arrows
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if not np.isnan(grid[i, j]):
                action = int(grid[i, j])
                ax.text(j + 0.5, i + 0.5, action_arrows[action], ha='center', va='center', fontsize=54)

    plt.show()


def plot_log_U(U_non_monotonic):
    """"
    Plots the delta U values in log scale w.r.t. the iteration number. The input must be a 1-d array with number of
    entries being equal to the number of iterations. Each entry is delta U which is the L2 norms of U - U*
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    y_values = np.log(U_non_monotonic)
    # Generating x-values from the index + 1
    x_values = np.arange(1, len(y_values) + 1)

    # Plotting
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
    ax.set_title('Iteration vs log U')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log U[s]')
    ax.yaxis.grid()



if __name__ == "__main__":
    U = np.array([ 0.22717064,  0.2298222 ,  0.31101657, -0.85377696,  0.29384167,
        0.47476549, -1.        ,  0.30962343,  0.43517645,  0.67124399,
        1.        ])

    # plot_gridworld_u(U)

    pi = np.array([0, 3, 0, 2, 0, 3, 0, 1, 1, 1, 0], dtype=int)
    # plot_policy(pi)

    plot(U, pi)
