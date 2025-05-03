"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

from agents.vi_agent import ViAgent

try:
    from world import Environment
    from agents.random_agent import RandomAgent
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
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


@staticmethod
def _neutral_reward_function(grid, agent_pos) -> float:
    """This is a very simple reward function.

    Args:
        grid: The grid the agent is moving on, in case that is needed by
            the reward function.
        agent_pos: The position the agent is moving to.

    Returns:
        A single floating point value representing the reward for a given
        action.
    """

    match grid[agent_pos]:
        case 0:  # Moved to an empty tile
            reward = -1
        case 1 | 2:  # Moved to a wall or obstacle
            reward = -5
            pass
        case 3:  # Moved to a target tile
            reward = 10
            # "Illegal move"
        case _:
            raise ValueError(f"Grid cell should not have value: {grid[agent_pos]}.",
                            f"at position {agent_pos}")
    return reward


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, reward_fn=_neutral_reward_function,
                          random_seed=random_seed)
        
        env.reset()
        # Initialize agent
        agent = ViAgent(gamma=0.9, grid_size=env.grid.shape, reward=env.reward_fn, grid=env.grid)
        
        # Always reset the environment to initial state
        state = env.reset()
        for _ in trange(iters):
            
            # Agent takes an action based on the latest observation and info.
            action = agent.take_action(state)

            # The action is performed in the environment
            state, reward, terminated, info = env.step(action)
            
            # If the final state is reached, stop.
            if terminated:
                break

            agent.update(state, reward, info["actual_action"])

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)