import numpy as np
from gymnasium.spaces import Discrete, MultiBinary

from mdp import MDP


class SwitchGridWorld(MDP):
    def __init__(self, gamma: float = 0.9):
        """
        Create the tensors T (A x S x S), R (A x S), and p0 (S).
        Example:
        For the implementation of the MDP base class please take a look at the
        mdp.py file in the folder.
        """

        state_size = 4
        action_size = 2

        p0 = np.zeros(state_size)
        p0[0] = 1.
        T_left = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        T_right = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ])

        T = np.array([T_left, T_right])

        R = np.zeros((action_size, state_size, state_size), dtype=float) - 1
        R[:, -1, -1] = 0

        # Initialize MDP
        super().__init__(T, R, p0, gamma=gamma)
        self.action_space = Discrete(action_size)
        self.observation_space = MultiBinary(2)
        self.current_state = None

    def reset(self) -> tuple[np.ndarray, dict]:
        self.current_state = 0
        return self.phi(self.current_state), {'state': self.current_state}

    def phi(self, state: int) -> np.ndarray:
        if self.terminal(state):
            return np.array([0, 1])
        return np.array([1, 0])

    def step(self, action: int) \
        -> tuple[int, float, bool, bool, dict]:
        prev_state = self.current_state
        self.current_state = np.random.choice(self.state_size,
                                              p=self.T[action, self.current_state])
        reward = self.R[action, prev_state, self.current_state]
        terminal = self.terminal(self.current_state)
        return self.phi(self.current_state), reward, terminal, False, {}
