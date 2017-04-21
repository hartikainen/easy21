import pickle
import argparse

from environment import Easy21Env
from agents import MonteCarloAgent, SarsaAgent, FunctionApproximationAgent
from vis import plot_Q

parser = argparse.ArgumentParser(
  description="Simple Reinforcement Learning Environment")

parser.add_argument("-a", "--agent", default="mc",
                    choices=['mc', 'sarsa', 'lfa'],
                    help=("Agent Type: "
                          "mc (monte carlo), "
                          "sarsa, "
                          "lfa (linear function approximation)"))
parser.add_argument("--num-episodes", default=1000, type=int,
                    help="Number of episodes")
parser.add_argument("-v", "--verbose", default=False, type=bool,
                    help="Verbose")
parser.add_argument("--lmbd", default=1, type=float, help="Lambda")
parser.add_argument("--gamma", default=1, type=float, help="Gamma")
parser.add_argument("--vis", default=False, type=bool,
                    help="Plot the value function")
parser.add_argument("--dump-q", default=False, type=bool,
                    help="Dump the Q values to file")


AGENTS = {
  "mc": MonteCarloAgent,
  "sarsa": SarsaAgent,
  "lfa": FunctionApproximationAgent
}


Q_DUMP_BASE_NAME = "Q_dump"
def dump_Q(Q, args):
  filename = ("./{}_{}_lambda_{}_gamma_{}_episodes_{}.pkl"
              "".format(Q_DUMP_BASE_NAME,
                        args.agent, args.lmbd, args.gamma, args.num_episodes))

  with open(filename, "wb") as f:
    pickle.dump(Q, f)


def main(args):
  agent_params = {
    "gamma": args.gamma,
    "lmbd": args.lmbd,
    "num_episodes": args.num_episodes
  }

  env = Easy21Env()
  agent = AGENTS[args.agent](env, **agent_params)

  Q = agent.learn()

  if args.vis:
    plot_Q(Q)

  if args.dump_q:
    dump_Q(Q, args)


if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
