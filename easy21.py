import pickle
import argparse
import numpy as np
from distutils.util import strtobool

from environment import Easy21Env, ACTIONS, DEALER_RANGE, PLAYER_RANGE
from agents import MonteCarloAgent, SarsaAgent, FunctionApproximationAgent
from vis import plot_V, plot_lambda_mse, plot_learning_curve
from utils import mse


def range_float_type(s):
  """ Custom range float type for arg parser
  """
  try:
    parts = list(map(float, s.split(",")))
    if len(parts) == 1:
      return parts
    elif len(parts) == 3:
      return np.arange(*parts)
  except:
    raise argparse.ArgumentTypeError(
      "range_float must be a string that, when split and parts then mapped to "
      "floats, can be passed to np.arange as arguments. E.g. '0,1.1,0.1'."
    )


def bool_type(x):
  return bool(strtobool(x))


parser = argparse.ArgumentParser(
  description="Simple Reinforcement Learning Environment")

parser.add_argument("-v", "--verbose", default=False, type=bool_type,
                    help="Verbose")

parser.add_argument("-a", "--agent", default="mc",
                    choices=['mc', 'sarsa', 'lfa'],
                    help=("Agent Type: "
                          "mc (monte carlo), "
                          "sarsa, "
                          "lfa (linear function approximation)"))
parser.add_argument("--num-episodes", default=1000, type=int,
                    help="Number of episodes")
parser.add_argument("--lmbd", default=[1.0], type=range_float_type, help="Lambda")
parser.add_argument("--gamma", default=1, type=float, help="Gamma")

parser.add_argument("--plot-v", default=False, type=bool_type,
                    help="Plot the value function")
parser.add_argument("--dump-q", default=False, type=bool_type,
                    help="Dump the Q values to file")
parser.add_argument("--plot-lambda-mse", default=False, type=bool_type,
                    help=("Plot mean-squared error compared to the 'true' Q "
                          "values obtained with monte-carlo"))
parser.add_argument("--plot-learning-curve", default=False, type=bool_type,
                    help=("Plot the learning curve of mean-squared error "
                          "compared to the 'true' Q values obtained from "
                          "monte-carlo against episode number"))


AGENTS = {
  "mc": MonteCarloAgent,
  "sarsa": SarsaAgent,
  "lfa": FunctionApproximationAgent
}

Q_DUMP_BASE_NAME = "Q_dump"
def dump_Q(Q, args):
  filename = ("./{}_{}_lambda_{}_gamma_{}_episodes_{}.pkl"
              "".format(Q_DUMP_BASE_NAME,
                        args["agent_type"], args["lmbd"],
                        args.get("gamma", None), args["num_episodes"]))

  print("dumping Q: ", filename)

  with open(filename, "wb") as f:
    pickle.dump(Q, f)


def get_agent_args(args):
  agent_type = args.agent
  agent_args = {
    "agent_type": agent_type,
    "num_episodes": args.num_episodes
  }

  if agent_type == "mc":
    return agent_args
  elif agent_type == "sarsa" or agent_type == "lfa":
    agent_args.update({
      key: getattr(args, key) for key in ["gamma"]
      if key in args
    })
    agent_args["save_error_history"] = getattr(
      args, "plot_learning_curve", False )

  return agent_args


Q_OPT_FILE = "./Q_opt.pkl"
def main(args):
  if args.plot_lambda_mse:
    with open(Q_OPT_FILE, "rb") as f:
      Q_opt = pickle.load(f)
      errors = {}

  env = Easy21Env()

  for i, lmbd in enumerate(args.lmbd):
    agent_args = get_agent_args(args)
    agent_args["lmbd"] = lmbd
    agent = AGENTS[args.agent](env, **agent_args)

    agent.learn()

    if args.dump_q:
      dump_Q(agent.Q, agent_args)

    if args.plot_v:
      plot_file = ("./vis/V_{}_lambda_{}_gamma_{}_episodes_{}.pdf"
                   "".format(agent_args["agent_type"],
                             lmbd,
                             args.gamma,
                             args.num_episodes))
      plot_V(agent.Q, save=plot_file)

    if args.plot_learning_curve:
      errors = agent.error_history
      plot_file = ("./vis/learning_curve_{}_gamma_{}_episodes_{}.pdf"
                  "".format(agent_args["agent_type"],
                            args.gamma,
                            args.num_episodes))
      plt = plot_learning_curve(errors, save=plot_file, agent_args=agent_args)

    if args.plot_lambda_mse:
      errors[lmbd] = mse(agent.Q, Q_opt)

  if args.plot_lambda_mse:
    plot_file = ("./vis/lambda_mse_{}_gamma_{}_episodes_{}.pdf"
                 "".format(agent_args["agent_type"],
                           args.gamma, args.num_episodes))
    errors_table = list(zip(*errors.items()))
    plot_lambda_mse(errors_table, save=plot_file)

if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
