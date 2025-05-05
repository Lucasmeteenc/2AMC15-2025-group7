"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

import numpy as np

try:
    from world import Environment
    from agents.random_agent import RandomAgent
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
    from agents.random_agent import RandomAgent
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
    p.add_argument("--num_episodes", type=int, default=5000,
                   help="Number of episodes to train for.")
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
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, num_episodes: int, fps: int,
         sigma: float, gamma: float, epsilon: float, min_epsilon: float,
         epsilon_decay: float, max_steps_per_episode: int, random_seed: int,
         early_stopping_patience: int):
    """
    Main loop for Monte Carlo Training.

    Args:
        grid_paths (list[Path]): List of paths to the grid files.
        no_gui (bool): If True, disables GUI rendering.
        num_episodes (int): Number of episodes to train for.
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

    for grid_path in grid_paths:
        print(f"\n--- Training on Grid: {grid_path.name} ---")

        # Set up the environment
        env = Environment(grid_path, no_gui=no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed)
        
        _ = env.reset() # Call reset once to load the grid
        grid_shape = env.grid.shape
        init_grid = np.copy(env.grid)
        num_actions = 4

        # Initialize agent
        agent = MonteCarloAgent(grid_shape=grid_shape,
                                num_actions=num_actions,
                                gamma=gamma,
                                initial_epsilon=epsilon,
                                min_epsilon=min_epsilon,
                                epsilon_decay=epsilon_decay)
        
        # Initialize variables for early stopping
        old_policy = np.zeros((grid_shape[0], grid_shape[1]), dtype=np.int32)
        same_policy_count = 0

        # Main training loop 
        for episode in trange(num_episodes, desc=f"Training on {grid_path.name}"):
            state = env.reset()
            terminated = False
            steps_in_episode = 0
            agent.episode_experience = []

            # Generate an episode of experience
            while not terminated and steps_in_episode < max_steps_per_episode:
                # Agent takes an action based on the current state and policy
                action = agent.take_action(state)

                # The action is performed in the environment
                next_state, reward, terminated, info = env.step(action)

                # Store the experience for this step
                agent.update(state, reward, action)

                # Move to the next state
                state = next_state
                steps_in_episode += 1

            # Once the episode is finished update Q-values
            agent.update_q_from_episode()

            # Update the exploration rate (epsilon)
            agent.update_epsilon()

            # Check for the early stopping condition
            new_policy = agent.get_policy()
            if np.array_equal(new_policy, old_policy):
                same_policy_count += 1
                if same_policy_count >= early_stopping_patience:
                    print(f"Policy converged after {episode} episodes.")
                    break
            else:
                same_policy_count = 0

            old_policy = new_policy.copy()

        print(f"\n--- Training completed for {grid_path.name} ---")
        
        # Evaluate the final policy
        eval_agent = MonteCarloAgent(grid_shape=grid_shape, num_actions=num_actions, gamma=gamma)
        eval_agent.Q = agent.Q # Copy the learned Q-values
        eval_agent.epsilon = 0 # Greedy policy for evaluation

        eval_agent.print_policy(init_grid)

        print(f"Evaluating agent on {grid_path.name}...")
        Environment.evaluate_agent(grid_path,
                                   eval_agent, # Use the greedy evaluation agent
                                   max_steps=max_steps_per_episode * 2, # Allow more steps for eval
                                   sigma=0, # Evaluate deterministically
                                   random_seed=random_seed)

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.num_episodes, args.fps, args.sigma,
         args.gamma, args.epsilon, args.min_epsilon, args.epsilon_decay,
         args.max_steps_per_episode, args.random_seed, args.early_stopping_patience)