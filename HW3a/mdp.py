import numpy as np


class MDP:

    def __init__(self, T: np.ndarray, R: np.ndarray, p0: np.ndarray, gamma: float = 0.9):
        """
        :param T: Transition matrix (A x S x S)
        :param R: Reward matrix (A x S)
        :param p0: Initial state distribution (S)
        :param gamma: Discount factor
        Note: Terminal states are denoted by states where for all actions,
        the probability of a transition to the same state is 1.
        """
        assert np.all(T.sum(axis=-1) == 1), "Transition matrix doesn't sum to 1 along last axis."
        assert np.all(p0.sum(axis=-1) == 1), "Initial state distribution doesn't sum to 1."
        assert gamma <= 1. and gamma >= 0, f"Gamma is {gamma}, not between 0 and 1."

        self.gamma = gamma
        self.T = T
        self.R = R
        self.p0 = p0

    @property
    def S(self) -> np.ndarray:
        return np.arange(self.state_size)

    @property
    def A(self) -> np.ndarray:
        return np.arange(self.action_size)

    def transitions(self, state: int, action: int) -> np.ndarray:
        transition_probs = self.T[action, state]
        rewards = np.zeros_like(transition_probs) + self.R[action, state]
        return np.column_stack((rewards, transition_probs))

    def terminal(self, state: int) -> bool:
        return np.all(self.T[:, state, state] == 1.)

    @property
    def state_size(self) -> int:
        return self.T.shape[-1]

    @property
    def action_size(self) -> int:
        return self.T.shape[0]

    def reset(self, rand_state: np.random.RandomState):
        raise NotImplementedError

    def step(self, state: int, action: int, rand_state: np.random.RandomState)\
            -> tuple[int, float, bool, bool, dict]:
        """
        Take a step in the MDP given a state and action.
        :param state: int specifying state index.
        :param action: int specifying action index.
        :param rand_state: numpy random state, for random sampling.
        :return: a tuple of (next_state, reward, terminal_signal, extra_info)
        """
        raise NotImplementedError


