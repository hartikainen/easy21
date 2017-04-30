# easy21
This repository implements the assignment requirements for the reinforcement learning course given by David Silver [1]. It implements a reinforcement learning environment and three different agents, namely monte-carlo, sarsa lambda, and linear function approximation, for simple card game called Easy21, presented in [1].


## Setting up the environment
To setup the python environment you need Python v3.x, pip, and virtualenv:
```
git clone https://github.com/hartikainen/easy21.git
cd easy21

pip install virtualenv
virtualenv .venv
source .venv/bin/activate

pip install -r requirements.txt
```


## Running the game
To run the game and test the agents, run the `easy21.py` file as follows:
```
python easy21.py [-h] [-v VERBOSE] [-a {mc,sarsa,lfa}]
                 [--num-episodes NUM_EPISODES] [--lmbd LMBD] [--gamma GAMMA]
                 [--plot-v PLOT_V] [--dump-q DUMP_Q]
                 [--plot-lambda-mse PLOT_LAMBDA_MSE]
                 [--plot-learning-curve PLOT_LEARNING_CURVE]
```
See `easy21.py` for more information about the running the game and testing the agents. All the agents are found in the `/agents` folder.


## Easy21 Environment
The Easy21 environment is implemented `Easy21Env` class found in `environment.py`. The environment keeps track of the game state (dealer card and sum of player cards), and exposes a `step` method, which, given an action (hit or stick), updates its state, and returns the observed state (in our case observation is equivalent to the game state) and reward.


## Monte-Carlo Control in Easy21
Monte-Carlo control for Easy21 is implemented in file `agents/monte_carlo.py`. The default implementation uses a time-varying scalar step-size α<sub>t</sub> = 1/N(s<sub>t</sub>, a<sub>t</sub>) and ε-greedy exploration strategy with ε<sub>t</sub> = N<sub>0</sub> / (N<sub>0</sub> + N(S<sub>t</sub>)), where N<sub>0</sub> = 100, N(s) is the number of times that state s has been visited, N(s,a) is the number of times that action a has been selected from state s, and t is the time-step.
The figure below presents the optimal value function V\* against the game state (player sum and dealer hand).

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/V_mc_1000000_episodes.png)


## TD Learning in Easy21
File `agents/sarsa.py` implements a Sarsa(λ) control for Easy21. It uses the same step-size and exploration schedules as the Monte-Carlo agent described in the previous section. The agent is tested with parameter values λ ∈ {0, 0.1, 0.2, ..., 1}, each ran for 20000 episodes. The first figure below present the learning curve, i.e. the mean-squared error vs. 'true' Q values against episode number, for each lambda. The next two figures plot the function V\* (same as in Monte-Carlo section) for λ=0.0 and λ=1.0.

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/lambda_mse_sarsa_gamma_1.0_episodes_20000.png)

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/V_sarsa_lambda_0.0_gamma_1.0_episodes_20000.png)

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/V_sarsa_lambda_1.0_gamma_1.0_episodes_20000.png)


### Bootstrapping in Easy21

As with any other situation, also in Easy21, bootstrapping reduces the variance of the learned policy, with the cost of increased bias. The Sarsa(λ) agent learns reasonable policy faster (i.e. in smaller number of episodes) than Monte-Carlo agent. I a game as simple as Easy21, however, it is feasible to run enough episodes for the Monte-Carlo agent to converge to the optimal unbiased policy.


### Bootstrapping in Easy21 vs Blackjack
The episodes in Easy21 on average last longer than in traditional Blackjack game because of the subtractive effect of red cards. Because of this, boostrapping is likely to be more useful in Easy21 than it would be in traditional Blackjack game.


## Linear Function Approximation in Easy21
File `agents/function_approximation.py` implements a value function approximator with coarse coding for Easy21, using binary feature vector φ(state,action) with 36 (3\*6\*2) features. Each binary feature takes value 1 if (state, action) lies in the cuboid of state-space corresponding to that feature, and the action corresponding to that feature, and 0 otherwise. The cuboids are defined in the variable `CUBOID_INTERVALS` in `agents/function_approximation.py`.

Similarly to the Sarsa(λ) in the previous section, we run tests with 20000 episodes for parameter values λ ∈ {0, 0.1, 0.2, ..., 1}, with constant step-size α=0.01 and exploration value ε=0.05. The figures below plot the learning curve for each lambda, and the function V\* for λ=0.0 and λ=1.0.

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/lambda_mse_lfa_gamma_1.0_episodes_20000.png)

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/V_lfa_lambda_0.0_gamma_1.0_episodes_20000.png)

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/V_lfa_lambda_1.0_gamma_1.0_episodes_20000.png)


### Notes for function approximation
Using approximation for the state and action space reduces the time and space complexity of the algorithm\*\*, due to the reduced number of variables, corresponding to the states and actions, needed to learn by the agent. However, this comes with the cost of reduced accuracy of the learned state-value function Q (and thus value function V and policy π). It seems like the overlapping regions in the cuboid intervals result in more extreme values in some states. This happens because each state in the expanded Q function approximation can be affected by multiple states and actions through the weights of function approximation.


### The effect of constant step-size α
One thing to notice is the effect of constant step-size used for the linear function approximation. Because the step-size is kept constant in the learning, some regions of the value function receive much less training than other parts. This results in incorrect value function in the extreme regions where the number of visits is small. This effect is tested with cuboid intervals corresponding to "identity" features, i.e. features that should result in exactly the same Q as learning with sarsa lambda. Even after 50k runs, the agent seems to have spots in the state space (low dealer and low player; low dealer and high player), where it doesn't match the sarsa results. This effect is presented in the figure below (for λ=0).

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/V_lfa_identity_features_static_alpha_lambda_0.0_gamma_1.0_episodes_50000.png)

When using dynamic step-size α, as in the Sarsa(λ) section above, we can see that this effect disappears, and the function approximation results in expected approximation, as shown in the figure below (again, for λ=0).

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/V_lfa_identity_features_dynamic_alpha_lambda_0.0_gamma_1.0_episodes_50000.png)

\*\* the actual running time is actually worse than with Sarsa(λ) because my function approximation implementation does not fully utilize numpy vectorization

[1] http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
