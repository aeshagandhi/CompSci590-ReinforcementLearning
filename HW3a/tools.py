from itertools import product
import math
from typing import Union, Callable

from gymnasium import Env, ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.core import ObsType, WrapperObsType
from IPython import display
import matplotlib.pyplot as plt
import numpy as np

from mdp import MDP

# Taken from https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/gym_lib.py#L41
CARTPOLE_MIN_VALS = np.array([-2.4, -5., -math.pi/12., -math.pi*2.])
CARTPOLE_MAX_VALS = np.array([2.4, 5., math.pi/12., math.pi*2.])
ACROBOT_MIN_VALS = np.array([-1., -1., -1., -1., -5., -5.])
ACROBOT_MAX_VALS = np.array([1., 1., 1., 1., 5., 5.])
MOUNTAINCAR_MIN_VALS = np.array([-1.2, -0.07])
MOUNTAINCAR_MAX_VALS = np.array([0.6, 0.07])



def functional_feature_value_determination(pi: np.ndarray,
                                           phi: np.ndarray,
                                           mdp: MDP):
    """
    Given a featurization scheme, returns the V values and Q values
    given a policy over the features.
    :param pi: Phi x A
    :param phi: S x Phi
    :param mdp: an MDP
    :return:
    """
    pi_state = phi @ pi
    occupancy = functional_get_occupancy(pi_state, mdp)

    p_pi_of_s_given_o = get_p_s_given_o(phi, occupancy)

    mdp_v_vals, mdp_q_vals = functional_value_determination(pi_state, mdp)

    # Q vals
    feature_q_vals = mdp_q_vals @ p_pi_of_s_given_o

    # V vals
    feature_v_vals = (feature_q_vals * pi.T).sum(0)

    return feature_v_vals, feature_q_vals


def functional_get_occupancy(pi_ground: np.ndarray, mdp: Union[MDP]):
    Pi_pi = pi_ground.transpose()[..., None]
    T_pi = (Pi_pi * mdp.T).sum(axis=0) # T^π(s'|s)

    # A*C_pi(s) = b
    # A = (I - \gamma (T^π)^T)
    # b = P_0
    A = np.eye(mdp.T.shape[-1]) - mdp.gamma * T_pi.transpose()
    b = mdp.p0
    return np.linalg.solve(A, b)


def get_p_s_given_o(phi: np.ndarray, occupancy: np.ndarray):
    repeat_occupancy = np.repeat(occupancy[..., None], phi.shape[-1], -1)

    # Q vals
    p_of_o_given_s = phi.astype(float)
    w = repeat_occupancy * p_of_o_given_s

    p_pi_of_s_given_o = w / (w.sum(axis=0) + 1e-10)
    return p_pi_of_s_given_o


def functional_value_determination(pi: np.ndarray, mdp: MDP):
    """
    Solves for V using linear equations.
    For all s, V_pi(s) = sum_s' sum_a[T(s'|s,a) * pi(a|s) * (R(s,a,s') + gamma * V_pi(s'))]
    pi: stochastic policy, of shape S x A
    mdp: MDP to evaluate policy on.
    """
    Pi_pi = pi.transpose()[..., None]
    T_pi = (Pi_pi * mdp.T).sum(axis=0) # T^π(s'|s)
    R_pi = (Pi_pi * mdp.T * mdp.R).sum(axis=0).sum(axis=-1) # R^π(s)

    # A*V_pi(s) = b
    # A = (I - \gamma (T^π))
    # b = R^π
    A = (np.eye(mdp.T.shape[-1]) - mdp.gamma * T_pi)
    b = R_pi
    v_vals = np.linalg.solve(A, b)

    R_sa = (mdp.T * mdp.R).sum(axis=-1)  # R(s,a)
    q_vals = (R_sa + (mdp.gamma * mdp.T @ v_vals))

    return v_vals, q_vals


def discounted_return(ep_rewards: np.ndarray[float], gamma) -> float:
    """
    Given an array of rewards we return the discounted return. Here, we dont
    expect you to account for terminal states being value zero. This is meant as
    a function that simply returns a sum of discounted rewards.
    """
    return np.dot(gamma ** np.arange(len(ep_rewards)), ep_rewards)


def rollout(env, pi: np.ndarray[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    We take as input an environment and a policy pi. We perform a single rollout.
    This rollout takes an action according to the policy pi and transitions according
    to the step function you defined above.
    """
    # Two lists for rewards and features
    ep_features = []
    ep_actions = []
    ep_rewards = []

    obs, _ = env.reset()
    ep_features.append(obs)
    terminal = False

    step = 0
    while not terminal:
        action = pi[env.current_state]
        obs, reward, terminal, _, _ = env.step(action)  #
        step += 1
        ep_features.append(obs)
        ep_actions.append(action)
        ep_rewards.append(reward)

    return np.array(ep_features), np.array(ep_rewards)


class FourierFeatures(ObservationWrapper):

    def __init__(self, env: Env,
                 min_vals, max_vals,
                 order: int = 2):
        super().__init__(env)
        self.order = order
        terms = product(range(order + 1), repeat=self.observation_space.shape[0])

        # Removing first iterate because it corresponds to the constant bias
        self.multipliers = np.array([list(map(int, x)) for x in terms][1:])
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.observation_space = Box(0, 1, shape=(self.multipliers.shape[0], ))

    def scale(self, values):
        shifted = values - self.min_vals
        if self.max_vals is None:
          return shifted

        return shifted / (self.max_vals - self.min_vals)

    def observation(self, observation: ObsType) -> WrapperObsType:
        scaled = self.scale(observation)
        return np.cos(np.pi * self.multipliers @ scaled)


def visualize_softmax_policy(env, theta,
                             softmax_fn: Callable,
                             total_steps: int = 1000):
    step = 0
    episode_returns = []
    while step < total_steps:
        terminal, truncated = False, False
        obs, info = env.reset()
        episode_return = 0

        img = plt.imshow(env.render())
        gamma_prev = 1.0

        while not terminal:
            img.set_data(env.render())
            display.display(plt.gcf())
            display.clear_output(wait=True)
            action = softmax_fn(theta, obs)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += gamma_prev*reward
            gamma_prev = gamma_prev*env.gamma
            step += 1

            if terminated or truncated:
                break
        episode_returns.append(episode_return)
    print("Visualization Ended.")
    return episode_returns


if __name__ == "__main__":
    import gymnasium as gym
    unwrapped_cartpole_env = gym.make('CartPole-v1', render_mode='rgb_array')
    cartpole_env = FourierFeatures(unwrapped_cartpole_env, CARTPOLE_MIN_VALS, CARTPOLE_MAX_VALS)
    obs, _ = cartpole_env.reset()
    print()
