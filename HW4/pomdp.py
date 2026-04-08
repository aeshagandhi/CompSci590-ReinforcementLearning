import numpy as np
from gymnasium.spaces import Discrete, MultiBinary

from mdp import MDP


class SimplePOMDP(MDP):
    def __init__(self, gamma: float = 0.5):
        """
        Create the tensors T (A x S x S), R (A x S), and p0 (S).
        Example:
        For the implementation of the MDP base class please take a look at the
        mdp.py file in the folder.
        """

        state_size = 4
        action_size = 2

        #GO EAST
        T_left = np.array([
            [0.9, 0.1, 0, 0],
            [0.9, 0, 0.1, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0.9, 0.1]
        ])
        #GO WEST
        T_right = np.array([
            [0.1, 0.9, 0, 0],
            [0.1, 0, 0.9, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0.1, 0.9]
        ])

        T = np.array([T_left, T_right])
        
        
        p0 = np.array([1/3, 1/3, 0.0, 1/3 ], dtype=float)

        R = np.zeros((action_size, state_size, state_size), dtype=float) - 1
        R[:, -2, -2] = 0

        # Initialize MDP
        super().__init__(T, R, p0, gamma=gamma)
        self.action_space = Discrete(action_size)
        self.observation_space = MultiBinary(1)
        self.current_state = None
        
    def observation_probability(self, o, s_prime, a) -> float:
        """
        Return the probability of observing o, given the agent is at s_prime
        and had previously taken the action a
        """
        
        if s_prime == 2:
            return float(o == 1)
        else:
            return float(o == 0)
            

    def reset(self) -> tuple[np.ndarray, dict]:
        self.current_state = np.random.choice(self.state_size,
                                              p=self.p0)
        return self.phi(self.current_state), {'state': self.current_state}

    def phi(self, state: int) -> np.ndarray:
        if self.terminal(state):
            return np.array([1])
        return np.array([0])
    
    
    def step(self, action: int) \
        -> tuple[int, float, bool, bool, dict]:
        prev_state = self.current_state
        self.current_state = np.random.choice(self.state_size,
                                              p=self.T[action, self.current_state])
        reward = self.R[action, prev_state, self.current_state]
        terminal = self.terminal(self.current_state)
        return self.phi(self.current_state), reward, terminal, False, {}

    
class TigerPOMDP(MDP):
    def __init__(self, gamma: float = 0.5):
        """
        Create the tensors T (A x S x S), R (A x S), and p0 (S).
        Example:
        For the implementation of the MDP base class please take a look at the
        mdp.py file in the folder.
        """

        state_size = 3
        action_size = 3
        
        #GO RIGHT
        T_right = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ])
        #GO LEFT
        T_left = np.array([
            [0.0, 0.0, 1.0], #Tiger Right
            [0.0, 0.0, 1.0], #Tiger Left
            [0.0, 0.0, 1.0]  #Terminal
        ])
        #LISTEN
        T_listen = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        T = np.array([T_right, T_left, T_listen])
        
        
        p0 = np.array([1/2, 1/2, 0.0], dtype=float)

        R = np.zeros((action_size, state_size), dtype=float)
        #R[2, 0] = -0.0 #Negative reward for listen action
        #R[2, 1] = -0.0
        R[0, 0] = -100.0 #Going Right on Tiger Right
        R[0, 1] = 10.0 #Going left on Tiger Right
        R[1, 0] = 10.0 #Going right on Tiger Left
        R[1, 1] = -100.0 #Going left on Tiger Left
        
        self.true_observation_probability = 0.85
        

        # Initialize MDP
        super().__init__(T, R, p0, gamma=gamma)
        self.action_space = Discrete(action_size)
        self.observation_space = MultiBinary(2)
        self.current_state = None
        
    def observation_probability(self, o, s_prime, a) -> float:
        """
        Return the probability of observing o, given the agent is at s_prime
        and had previously taken the action a
        """
        if o[2] == 1.0 and s_prime == 2:
            return 1.0
        elif o[s_prime] == 1.0 and a == 2:
            return self.true_observation_probability
        elif o[s_prime] == 0.0 and a == 2:
            return 1.0 - self.true_observation_probability
        
        return 0.0
        

    def reset(self) -> tuple[np.ndarray, dict]:
        self.current_state = np.random.choice(self.state_size,
                                              p=self.p0)
        return [0.0, 0.0, 0.0], {'state': self.current_state}
    
    def describe_observation(self, observation: np.ndarray) -> str:
        description = "unkown"
        if observation[0] == 1:
            description = "Roar right"
        elif observation[1] == 1:
            description = "Roar left"
        elif observation[2] == 1:
            description = "Terminal state"
        
        return description

    def phi(self, state: int) -> np.ndarray:
        """
        [1, 0, 0] -> Roar heard from right
        [0, 1, 0] -> Roar heard from left
        [0, 0, 1] -> Terminal observation
        """
        
        #Initialise the observation
        observation = np.zeros(3, dtype=float)
        if self.terminal(state):
            observation[self.current_state] = 1.0
        elif np.random.uniform() < self.true_observation_probability:
            #If the observation is correct
            observation[self.current_state] = 1.0
        else:
            #If the observation is incorrect
            observation[1 - self.current_state] = 1.0
        
        return observation
    
    
    def step(self, action: int) \
        -> tuple[int, float, bool, bool, dict]:
        prev_state = self.current_state
        self.current_state = np.random.choice(self.state_size,
                                              p=self.T[action, self.current_state])
        reward = self.R[action, prev_state]
        terminal = self.terminal(self.current_state)
        return self.phi(self.current_state), reward, terminal, False, {}
    

START_UP = 0
START_DOWN = 1
JUNCTION_UP = -3
JUNCTION_DOWN = -2
TERMINAL = -1


class TMaze(MDP):
    def __init__(self, n: int = 2,
                 discount: float = 0.9,
                 good_term_reward: float = 4.0,
                 bad_term_reward: float = -0.1):
        """
        Return T, R, gamma, p0 and phi for tmaze, for a given corridor length n

                                +---+
              X                 | G |     S: Start
            +---+---+---+   +---+---+     X: Goal Indicator
            | S |   |   |...|   | J |     J: Junction
            +---+---+---+   +---+---+     G: Terminal (Goal)
              0   1   2       n | T |     T: Terminal (Non-Goal)
                                +---+
        """
        n_states = 2 * n # Corridor
        n_states += 2 # Start
        n_states += 2 # Junction
        n_states += 1 # Terminal state (includes both goal/non-goal)

        T_up = np.eye(n_states, n_states)
        T_down = T_up.copy()
        T_up[[TERMINAL, JUNCTION_DOWN, JUNCTION_UP], [TERMINAL, JUNCTION_DOWN, JUNCTION_UP]] = 0
        T_down[[TERMINAL, JUNCTION_DOWN, JUNCTION_UP], [TERMINAL, JUNCTION_DOWN, JUNCTION_UP]] = 0

        # If we go up or down at the junctions, we terminate
        T_up[[JUNCTION_DOWN, JUNCTION_UP], [TERMINAL, TERMINAL]] = 1
        T_down[[JUNCTION_DOWN, JUNCTION_UP], [TERMINAL, TERMINAL]] = 1

        T_left = np.zeros((n_states, n_states))
        T_right = T_left.copy()

        # At the leftmost and rightmost states we transition to ourselves
        T_left[[START_UP, START_DOWN], [START_UP, START_DOWN]] = 1
        T_right[[JUNCTION_DOWN, JUNCTION_UP], [JUNCTION_DOWN, JUNCTION_UP]] = 1

        # transition to -2 (left) or +2 (right) index
        all_nonterminal_idxes = np.arange(n_states - 1)
        T_left[all_nonterminal_idxes[2:], all_nonterminal_idxes[:-2]] = 1
        T_right[all_nonterminal_idxes[:-2], all_nonterminal_idxes[2:]] = 1

        T = np.array([T_up, T_down, T_right, T_left])

        # Specify last state as terminal
        T[:, TERMINAL, TERMINAL] = 1

        R_left = np.zeros((n_states, n_states))
        R_right = R_left.copy()

        R_up = R_left.copy()
        R_down = R_up.copy()

        # If rewarding state is north
        R_up[JUNCTION_UP, TERMINAL] = good_term_reward
        R_down[JUNCTION_UP, TERMINAL] = bad_term_reward

        # If rewarding state is south
        R_up[JUNCTION_DOWN, TERMINAL] = bad_term_reward
        R_down[JUNCTION_DOWN, TERMINAL] = good_term_reward

        R = np.array([R_up, R_down, R_right, R_left])

        # Initialize uniformly at random between north start and south start
        p0 = np.zeros(n_states)
        p0[:2] = 0.5

        # Observation function with have 5 possible obs:
        # start_up, start_down, corridor, junction, and terminal
        phi = np.zeros((n_states, 4 + 1))

        # The two start states have observations of their own
        # (up or down)
        phi[0, 0] = 1
        phi[1, 1] = 1

        # All corridor states share the same observation (idx 2)
        phi[2:-3, 2] = 1

        # All junction states share the same observation (idx 3)
        phi[-3:-1, 3] = 1

        # we have a special termination observations
        phi[-1, 4] = 1

        super().__init__(T, R, p0, gamma=discount)
        self.phi_matrix = phi
        self.action_space = Discrete(4)
        self.observation_space = MultiBinary(phi.shape[-1])
        self.current_state = None

    def reset(self) -> tuple[np.ndarray, dict]:
        self.current_state = np.random.choice(self.state_size,
                                              p=self.p0)
        obs = self.phi(self.current_state)
        return obs, {'state': self.current_state}

    def phi(self, state: int):
        obs_idx = np.random.choice(self.phi_matrix.shape[-1], p=self.phi_matrix[state])
        return np.eye(self.phi_matrix.shape[-1])[obs_idx]

    def step(self, action: int) \
            -> tuple[int, float, bool, bool, dict]:
        prev_state = self.current_state
        self.current_state = np.random.choice(self.state_size,
                                              p=self.T[action, self.current_state])
        obs = self.phi(self.current_state)
        reward = self.R[action, prev_state, self.current_state]
        terminal = self.terminal(self.current_state)
        return obs, reward, terminal, False, {}
