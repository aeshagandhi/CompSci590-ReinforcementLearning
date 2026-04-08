import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from mdp import MDP

from tools import add_dims_and_convert, one_hot_action, eval_rnn_q, discounted_return


def eps_greedy_base(q_values: np.ndarray, epsilon: float = 0.1):
    """
    Sample an epsilon-greedy action.
    """

    index = np.argmax(q_values)
    pi = np.zeros_like(q_values)
    pi[index] = (1 - epsilon)
    pi += epsilon / pi.shape[0]
    chosen_action = np.random.choice(pi.shape[0], p=pi)
    
    return chosen_action

def q_learning(env: MDP, 
               alpha: float = 0.01,
               epsilon: float = 0.1,
               epsilon_decay: float = 0.0,
               decay_steps: float = 0,
               n_samples: int = 1000,
               print_logs=False,
               truncation_threshold=np.inf,
               ) -> np.ndarray:
    """
    Q-learning with linear value function approximation, quadratic features, and epsilon-greedy exploration.
    """
    w = np.ones((env.action_size, env.features_size))

    step = 0
    actions_dict = {0: 0, 1: 0, 2:0}
    while step < n_samples:

        obs, _, = env.featurized_reset()
        q_vals = (np.expand_dims(obs, axis=0) * w).sum(axis=-1)
        #print(q_vals)
        if print_logs:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RESET")
            print("Underlying state: " + str(env.pomdp.current_state))
        
        terminal = False
        episode_steps = 0
        

        while not terminal and episode_steps <= truncation_threshold:
            action = eps_greedy_base(q_vals, epsilon=epsilon)
            next_obs, reward, terminal, truncated, variables_dict = env.featurized_step(action)
            actions_dict[action] += 1
            episode_steps += 1
            
            
            discount = env.gamma if not terminal else 0
            next_q_vals = (np.expand_dims(next_obs, axis=0) * w).sum(axis=-1)
            
            
            target = reward + discount * np.max(next_q_vals)
            if print_logs:
                print("*************************")
                print(f"Action: {action}")
                print(f"Observation: {obs}")
                print("Next q vals: " + str(next_q_vals))
                print("Q vectors: " + str(w))
                print("Next obs: " + str(next_obs))
                print("Reward: " + str(reward))
                print("Target: " + str(target))
                print("Underlying state: " + str(env.pomdp.current_state))
                print("Underlying observation:" + str(variables_dict["observation"]))
                print("*************************")
            w[action] = w[action] + alpha * (target - q_vals[action]) * obs
            step += 1
            if decay_steps != 0 and step % decay_steps == 0:
                epsilon = epsilon*epsilon_decay
            if terminal or truncated or step >= n_samples:
                break
            
            obs = next_obs
            q_vals = next_q_vals
    print(actions_dict)

    return w
