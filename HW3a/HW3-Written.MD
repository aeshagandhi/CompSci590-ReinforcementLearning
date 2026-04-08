# HW 3: Policy Gradients and Actor-Critic Algorithms

Name: **Aesha Gandhi**


### 1.1

In this problem, if the policy goes left, the agent stays at state 0 forever and if the policy goes right, the agent goes to state 1, then tries to go right which bounces back left to state 0. So either way, the agent can never end up at the goal state. The reward at every time step is -1, so this becomes a geometric series summing up the discounted -1 rewards. Using properties of geometric series, this sums to 10.
$$
V = \sum_{t=0}^{\infty} \gamma^t \cdot (-1)
= \frac{-1}{1 - \gamma}
= \frac{-1}{0.1}
= -10
$$


### 1.2 
(i = 0, j = 3) tells us the probability of going from the start state (state 0) and reaching the goal state (state 3 since it is terminal and is the only absorbing state) in 10 steps. The always going right policy gives probability 0 for this since it never reaches the goal. While using the non zero probabilities shows that the higher probability then means less episodes needed to reach the goal, less -1 penalties, and thus higher value. This shows that the stochastic policies can get the higher values.