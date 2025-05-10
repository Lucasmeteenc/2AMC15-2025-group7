"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import random

try:
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.vi_agent import ViAgent
    from agents.mc_agent import MonteCarloAgent
    from agents.q_learning_agent import QLearningAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.vi_agent import ViAgent
    from agents.mc_agent import MonteCarloAgent
    from agents.q_learning_agent import QLearningAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iters", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--agent", type=str, default="vi",
                   help="Agent selection: vi (Value Iteration), mc (On Policy Monte Carlo), ql (Q-Learning).")
    
    # Monte Carlo specific parameters
    p.add_argument("--max_steps_per_episode", type=int, default=500, # Safety limit
                   help="Maximum steps allowed per episode.")
    p.add_argument("--gamma", type=float, default=0.9, # Discount factor
                   help="Discount factor gamma.")
    p.add_argument("--epsilon", type=float, default=1.0, # Initial Epsilon
                   help="Initial exploration rate epsilon.")
    p.add_argument("--min_epsilon", type=float, default=0.05, # Minimum Epsilon
                   help="Minimum exploration rate epsilon.")
    p.add_argument("--epsilon_decay", type=float, default=0.9995, # Epsilon decay rate
                   help="Epsilon decay rate per episode.")
    p.add_argument("--early_stopping_patience_mc", type=int, default=250,
                   help="Amount of episodes with the same policy that triggers early stopping.")
    
    # Q-Learning specific parameters
    p.add_argument("--early_stopping_patience_ql", type=int, default=50,
                   help="Amount of episodes with the same policy that triggers early stopping.")
    return p.parse_args()


def run_train_loop(agent, grid, no_gui, iters, fps, sigma, gamma, epsilon, min_epsilon, epsilon_decay, max_steps_per_episode, early_stopping_patience_mc, early_stopping_patience_q):
    # Define random seed such that each loop is different.
    random_seed = random.randint(-1000000000, 1000000000)
    
    env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)

    # Always reset the environment to initial state
    _ = env.reset()

    if agent == "vi":
        print("Using Value Iteration agent")

        agent = ViAgent(grid_name=grid, gamma=gamma, grid_size=env.grid.shape, reward=env.reward_fn, grid=env.grid, sigma=sigma)

        agent.train(env)

    elif agent == "mc":
        print("Using On Policy Monte Carlo agent")

        gamma = gamma
        epsilon = epsilon
        min_epsilon = min_epsilon
        epsilon_decay = epsilon_decay
        max_steps_per_episode = max_steps_per_episode
        early_stopping_patience = early_stopping_patience_mc

        grid_shape = env.grid.shape

        agent = MonteCarloAgent(grid_shape=grid_shape,
                                grid_name=grid,
                                gamma=gamma,
                                initial_epsilon=epsilon,
                                min_epsilon=min_epsilon,
                                epsilon_decay=epsilon_decay,
                                stochasticity=sigma,
                                max_steps_per_episode=max_steps_per_episode)
    
        agent.train(env, iters, max_steps_per_episode, early_stopping_patience)

        # Set the exploration rate to 0 for evaluation
        agent.epsilon = 0
    
    elif agent == "ql":
        print("Using Q-Learning agent")

        agent = QLearningAgent(env.grid, 
                                gamma=gamma, 
                                grid_name=grid,
                                stochasticity=sigma,
                                initial_epsilon=epsilon,
                                max_steps_per_episode=max_steps_per_episode)

        agent.train(env, iters, max_steps_per_episode, early_stopping_patience_q)

        # Set the exploration rate to 0 for evaluation
        agent.epsilon = 0
    
    else:
        raise ValueError(f"Unknown agent type: {agent}")
    
    # evaluate (not needed currently)
    # Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


def main_dispatcher():
    args = parse_args()

    no_gui = args.no_gui
    sigma = args.sigma
    fps = args.fps
    random_seed = args.random_seed
    agent = args.agent
    iters = args.iters
    gamma = args.gamma
    sigma = args.sigma

    epsilon = args.epsilon
    min_epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    max_steps_per_episode = args.max_steps_per_episode

    early_stopping_patience_ql = args.early_stopping_patience_ql
    early_stopping_patience_mc = args.early_stopping_patience_mc

    agents = ["vi", "ql", "mc"]
    sigmas = [0.0, 0.1, 0.3]
    # gammas = [0.4, 0.9]
    gammas = []
    # start_epsilons = [1.0, 0.7, 0.3, 0.1]
    start_epsilons = []
    # episode_lengths = [100, 250, 500, 1000]
    episode_lengths = []

    for run in range(3):
        for agent in agents:
            for grid in args.GRID:
                # Stochasity
                for mod_sigma in sigmas:
                    run_train_loop(agent, grid, no_gui, iters, fps, mod_sigma, gamma, epsilon, min_epsilon, epsilon_decay, max_steps_per_episode, early_stopping_patience_mc, early_stopping_patience_ql)
                
                # Dicounted reward
                for mod_gamma in gammas:
                    run_train_loop(agent, grid, no_gui, iters, fps, sigma, mod_gamma, epsilon, min_epsilon, epsilon_decay, max_steps_per_episode, early_stopping_patience_mc, early_stopping_patience_ql)

                if agent != "vi":
                    # Starting epsilon. Is still decayed using the same epsilon_decay 
                    for mod_epsilon in start_epsilons:
                        run_train_loop(agent, grid, no_gui, iters, fps, sigma, gamma, mod_epsilon, min_epsilon, epsilon_decay, max_steps_per_episode, early_stopping_patience_mc, early_stopping_patience_ql)

                    # Episode length (both MC and Q-learning, although less efficient for Q-learning.)
                    for mod_episode_length in episode_lengths:
                        run_train_loop(agent, grid, no_gui, iters, fps, sigma, gamma, epsilon, min_epsilon, epsilon_decay, mod_episode_length, early_stopping_patience_mc, early_stopping_patience_ql)

                    # TODO learning rate.


if __name__ == '__main__':
    main_dispatcher()