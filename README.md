# easy21
This repository implements the assignment requirements for the reinforcement learning course given by David Silver [1]. It implements a reinforcement learning environment and three different agents, namely monte-carlo, sarsa lambda, and linear function approximation, for simple card game called Easy21 presented in [1].

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
Monte-Carlo control for Easy21 is implemented in file `agents/monte_carlo.py`. The default implementation uses a time-varying scalar step-size of 1/N(s_t, a_t) and epsilon-greedy exploration strategy with epsilon_t = N_0 / (N_0 + N(S_t)), where N_0 = 100, N(s) is the number of times that state s has been visited, and N(s,a) is the number of times that action a has been selected from state s.
The figure below presents the optimal value function V* against the game state (player sum and dealer hand).

![alt text](https://github.com/hartikainen/easy21/blob/master/vis/V_mc_1000000_episodes.png)

## TODO:
The function approximation agent seems to have a bug. I tested running with "identity" features, i.e. features that should result in exactly the same Q as learning with sarsa lambda. However, even after 50k runs, the agent seems to have spots in the state space (low dealer and low player; low dealer and high player), where it doesn't match the sarsa results.


[1] http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
