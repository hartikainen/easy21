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

## TODO:
The function approximation agent seems to have a bug. I tested running with "identity" features, i.e. features that should result in exactly the same Q as learning with sarsa lambda. However, even after 50k runs, the agent seems to have spots in the state space (low dealer and low player; low dealer and high player), where it doesn't match the sarsa results.


[1] http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
