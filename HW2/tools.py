import numpy as np
import matplotlib.pyplot as plt
from IPython import display


plt.rc('font', size=30)  # controls default text sizes
plt.rc('axes', titlesize=25)  # fontsize of the axes title
plt.rc('axes', labelsize=25)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=17)  # fontsize of the tick labels
plt.rc('ytick', labelsize=17)  # fontsize of the tick labels
plt.rc('legend', fontsize=20)  # legend fontsize
plt.rc('figure', titlesize=30)
plt.tight_layout()


def plot_gridworld_u(value_vector):
    # Create a grid full of NaNs
    grid = np.full((3, 4), np.nan)

    # Fill the grid with the values, skipping the obstacle position (1,1)
    grid[0, :] = value_vector[:4]  # First row
    grid[1, :] = value_vector[4:8]  # Second row, after obstacle
    grid[2, :] = value_vector[8:-1]  # Third row
    grid[1, 1] = np.nan

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


def get_policy_vector(w, env):
    policy_vector = [np.argmax((w * env.phi(s)[None, :]).sum(axis=-1)) for s in env.S]
    return policy_vector

def plot_greedy_policy(w, env, ax=None):
    """
    Plots the policy for a gridworld with dividers between each state for improved visualization.

    Parameters:
    - grid_shape: tuple, the shape of the grid (rows, columns)
    - policy_vector: list, the deterministic policy actions for each state, indexed by the state number.
                     Actions are encoded as [0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT']
    """

    policy_vector = get_policy_vector(w, env)

    grid_shape = (3, 4)
    grid = np.full(grid_shape, np.nan)

    # Fill the grid with the values, skipping the obstacle position (1,1)
    grid[0, :] = policy_vector[:4]  # First row
    grid[1, :] = policy_vector[4:8]  # Second row
    grid[2, :] = policy_vector[8:-1]  # Third row
    grid[1, 1] = np.nan

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

def plot_smooth_U(U_vals) : 
    N = 100
    smooth_rets = np.convolve(U_vals, np.ones(N)/N, mode='valid')
    x = [i + 1 for i in range(len(smooth_rets))]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    y_values = smooth_rets
    # Generating x-values from the index + 1
    x_values = np.arange(1, len(y_values) + 1)

    # Plotting
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
    ax.set_title('Iteration vs Return')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Returns')
    ax.yaxis.grid()

def obtain_cartpole_return_estimate(env, w, total_steps: int = 1000):
    step = 0
    episode_returns = []
    while step < total_steps:
        terminal, truncated = False, False
        obs, info = env.featurized_reset()
        q_vals = (np.expand_dims(obs, axis=0) * w).sum(axis=-1)
        episode_return = 0

        while not terminal:
            action = np.argmax(q_vals)

            obs, reward, terminated, truncated, info = env.featurized_step(action)
            episode_return += reward
            step += 1
            q_vals = (np.expand_dims(obs, axis=0) * w).sum(axis=-1)

            if terminated or truncated:
                break
        episode_returns.append(episode_return)
    
    return episode_returns
    
    
def visualize_q_weight_policy(env, w,
                              total_steps: int = 1000):
    step = 0
    episode_returns = []
    while step < total_steps:
        terminal, truncated = False, False
        obs, info = env.featurized_reset()
        q_vals = (np.expand_dims(obs, axis=0) * w).sum(axis=-1)
        episode_return = 0

        img = plt.imshow(env.render())
        gamma_prev = 1.0

        while not terminal:
            img.set_data(env.render())
            display.display(plt.gcf())
            display.clear_output(wait=True)
            action = np.argmax(q_vals)

            obs, reward, terminated, truncated, info = env.featurized_step(action)
            episode_return += gamma_prev*reward
            gamma_prev = gamma_prev*env.gamma
            step += 1
            q_vals = (np.expand_dims(obs, axis=0) * w).sum(axis=-1)

            if terminated or truncated:
                break
        episode_returns.append(episode_return)
    print("Visualization Ended.")
    return episode_returns

def visualize_random(env, total_steps: int = 100):
    step = 0
    while step < total_steps:
        terminal, truncated = False, False
        observation, info = env.featurized_reset()

        img = plt.imshow(env.render())

        while not terminal:
            img.set_data(env.render())
            display.display(plt.gcf())
            display.clear_output(wait=True)
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.featurized_step(action)
            step += 1
            if terminated or truncated:
                break
    print("Visualization Ended.")

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
    grid[2, :] = policy_vector[7:-1]  # Third row

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
