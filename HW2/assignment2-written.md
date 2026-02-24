# Section 2

## 2.1

In the optimal policy, states 2 and 3 have values 0.0. This is because they are never visited during the rollouts performed starting from the initial state. MC Policy Evaluation creates estimates of each state's value by averaging the returns observed when the state appears in an episode, so if a state is never encountered its value defaults to 0. Under the optimal policy, the agent's path reaches the +1 terminal state and avoids the risky regions near the -1 terminal state, so the path doesn't pass through states 2 and 3. Thus, these states never get visited during sampling and receive values of 0.0.

## 2.2

The values remain almost the same because introducing the terminal state only changes when the ±1 reward is received, not the overall return structure of the MDP. In the previous MDP, for states $s \in \{11,7\}$ we had $V(s)=\mathbb{E}[r + \gamma V(s')]=\mathbb{E}[r]$ since termination implied $V(s')=0$, whereas now $V(s)=\mathbb{E}[r + \gamma V(12)]$ and since $V(12)=0$, this either remains $\mathbb{E}[r]$ or becomes $\gamma \mathbb{E}[r]$ if the reward is delivered one step later on the transition to state 12.

# Section 3

## 3.1

$$\overline{VE}(w) = \mathbb{E}_\pi \left[ \left( \sum_i \gamma^i R_i - \hat{V}(S,w) \right)^2 \right]$$

Let $G = \sum_i \gamma^i R_i$ be the Monte Carlo return. Then:

$$\overline{VE}(w) = \mathbb{E}_\pi \left[ (G - \hat{V}(S,w))^2 \right]$$

Taking the gradient:

$$\nabla_w \overline{VE}(w) = \mathbb{E}_\pi \left[ 2 (G - \hat{V}(S,w)) \nabla_w (G - \hat{V}(S,w)) \right]$$

$$= \mathbb{E}_\pi \left[ 2 (G - \hat{V}(S,w)) (- \nabla_w \hat{V}(S,w)) \right]$$

Since $\hat{V}(S,w) = w^T \phi(S)$, we have $\nabla_w \hat{V}(S,w) = \phi(S)$. Therefore:

$$\nabla_w \overline{VE}(w) = \mathbb{E}_\pi \left[ -2 (G - \hat{V}(S,w)) \phi(S) \right]$$

Applying stochastic gradient descent with a single sample:

$$w_{t+1} = w_t - \alpha \left[ -2 (G - \hat{V}(S,w_t)) \phi(S) \right]$$

$$w_{t+1} = w_t + 2\alpha (G - \hat{V}(S,w_t)) \phi(S)$$

Absorbing the constant 2 into the learning rate:

$$w_{t+1} = w_t + \alpha (G - \hat{V}(S,w_t)) \phi(S)$$

## 3.2

The bottom-right states now have values because feature-based linear approximation uses shared weights across states, so updating from any visited state indirectly updates the estimated value of all other states including unvisited ones. In the previous index-based MC, states that were never visited stayed at 0 because there was no connection between states, but with feature vectors states share structure so weight updates generalize.

## 3.3

Because states share features, the weights get pulled in multiple directions at once. The -1 state has x-position 4, but so does the +1 state, so the weights must balance representing both, meaning the -1 state doesn't get represented perfectly and gets dragged upward toward 0. Essentially the learned weights have to fit all states simultaneously as a compromise.

## 3.4

If using one-hot vectors for state indices, online_mc is equivalent to the standard tabular Monte Carlo method since each state's update is completely independent — only the single weight corresponding to the current state gets updated, which is exactly like maintaining a separate running average of returns per state with no generalization across states.

# Section 6

## 6.1

The 3D plots show that the relationship between Q-values and state variables is nonlinear, but linear function approximation can only represent linear relationships, so the model cannot capture the true Q-function or learn the optimal policy using only the vanilla features.

## 6.2

I represented the state with the feature vector:

$$\phi_s = [1,\ \text{pos},\ \text{vel},\ \text{angle},\ \text{angle\_vel},\ \text{angle}^2,\ \text{angle\_vel}^2,\ \text{angle} \cdot \text{angle\_vel},\ \text{pos} \cdot \text{angle},\ \text{vel} \cdot \text{angle\_vel}]$$

This adds quadratic and interaction terms to the raw observations.

## 6.3

From the 3D plots, the Q-value surfaces are clearly parabolic with respect to angle and angular velocity, peaking at zero and dropping off symmetrically, which motivated the quadratic features $\text{angle}^2$ and $\text{angle\_vel}^2$ as well as the interaction term $\text{angle} \cdot \text{angle\_vel}$ to capture their joint dependency. Position and velocity contribute more linearly but interact with the angular features — seen as folds and saddle shapes in the plots — so cross terms like $\text{pos} \cdot \text{angle}$ and $\text{vel} \cdot \text{angle\_vel}$ were added to give the linear model enough expressiveness to approximate these nonlinear relationships.