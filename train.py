"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path

try:
    from world import Environment
    from agents.mc_agent import MonteCarloAgent
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
    from agents.mc_agent import MonteCarloAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer - Monte Carlo.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=300,
                   help="Frames per second (only if GUI enabled).")
    p.add_argument("--iter", type=int, default=5000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    
    # Monte Carlo specific parameters
    p.add_argument("--max_steps_per_episode", type=int, default=500, # Safety limit
                   help="Maximum steps allowed per episode.")
    p.add_argument("--gamma", type=float, default=0.99, # Discount factor
                   help="Discount factor gamma.")
    p.add_argument("--epsilon", type=float, default=1.0, # Initial Epsilon
                   help="Initial exploration rate epsilon.")
    p.add_argument("--min_epsilon", type=float, default=0.05, # Minimum Epsilon
                   help="Minimum exploration rate epsilon.")
    p.add_argument("--epsilon_decay", type=float, default=0.9995, # Epsilon decay rate
                   help="Epsilon decay rate per episode.")
    p.add_argument("--early_stopping_patience", type=int, default=250,
                   help="Amount of episodes with the same policy that triggers early stopping.")
    
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, gamma: float, epsilon: float, min_epsilon: float,
         epsilon_decay: float, max_steps_per_episode: int, random_seed: int,
         early_stopping_patience: int):
    """
    Main loop for Monte Carlo Training.

    Args:
        grid_paths (list[Path]): List of paths to the grid files.
        no_gui (bool): If True, disables GUI rendering.
        iters (int): Number of episodes to train for.
        fps (int): Frames per second for GUI rendering.
        sigma (float): Sigma value for the stochasticity of the environment.
        gamma (float): Discount factor.
        epsilon (float): Initial exploration rate.
        min_epsilon (float): Minimum exploration rate.
        epsilon_decay (float): Epsilon decay rate per episode.
        max_steps_per_episode (int): Maximum steps allowed per episode.
        random_seed (int): Random seed value for the environment.
        early_stopping_patience (int): Amount of episodes with the same policy that triggers early stopping.
    """

    for grid in grid_paths:
        # Set up the environment
        env = Environment(grid, no_gui=no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed)
        
        _ = env.reset() # Call reset once to load the grid
        grid_shape = env.grid.shape

        # Initialize agent
        agent = MonteCarloAgent(grid_shape=grid_shape,
                                gamma=gamma,
                                initial_epsilon=epsilon,
                                min_epsilon=min_epsilon,
                                epsilon_decay=epsilon_decay)
        
        agent.train(env, iters, max_steps_per_episode, early_stopping_patience)

        # Evaluate agent with a greedy policy
        print(f"Evaluating agent on {grid.name}...")

        agent.epsilon = 0
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iters, args.fps, args.sigma,
         args.gamma, args.epsilon, args.min_epsilon, args.epsilon_decay,
         args.max_steps_per_episode, args.random_seed, args.early_stopping_patience)