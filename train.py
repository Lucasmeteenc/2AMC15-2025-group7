"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

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
    # Default parameters
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
    p.add_argument("--iters", type=int, default=100000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor gamma.")
    
    # Agent selection
    p.add_argument("--agent", type=str, default="vi",
                   help="Agent selection: vi (Value Iteration), mc (On Policy Monte Carlo), ql (Q-Learning).")
    
    # Monte Carlo and Q-Learning shared parameters
    p.add_argument("--max_steps_per_episode", type=int, default=5000,
                   help="Maximum steps allowed per episode.")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="Initial exploration rate epsilon.")
    p.add_argument("--min_epsilon", type=float, default=0.0005,
                   help="Minimum exploration rate epsilon.")
    p.add_argument("--epsilon_decay", type=float, default=0.9999,
                   help="Epsilon decay rate per episode.")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Initial learning rate alpha.")
    p.add_argument("--min_alpha", type=float, default=0.00005,
                   help="Minimum learning rate alpha.")
    p.add_argument("--alpha_decay", type=float, default=0.9999,
                   help="Alpha decay rate per episode.")
    
    # Specific parameters
    p.add_argument("--early_stopping_patience_mc", type=int, default=1000,
                   help="Amount of episodes with the same policy that triggers early stopping.")
    p.add_argument("--early_stopping_patience_ql", type=int, default=50,
                   help="Amount of episodes with the same policy that triggers early stopping.")
    return p.parse_args()

def main_dispatcher():
    args = parse_args()

    no_gui = args.no_gui
    sigma = args.sigma
    fps = args.fps
    random_seed = args.random_seed
    agent = args.agent
    iters = args.iters

    for grid in args.GRID:

        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)
        
        # Always reset the environment to initial state
        _ = env.reset()

        if agent == "vi":
            print("Using Value Iteration agent")

            agent = ViAgent(gamma=args.gamma, grid_size=env.grid.shape, reward=env.reward_fn, grid=env.grid, sigma=sigma)

            agent.train(env)

        elif agent == "mc":
            print("Using On Policy Monte Carlo agent")

            agent = MonteCarloAgent(grid_shape = env.grid.shape,
                                    grid_name = grid,
                                    gamma = args.gamma,
                                    initial_epsilon = args.epsilon,
                                    min_epsilon = args.min_epsilon,
                                    epsilon_decay = args.epsilon_decay,
                                    initial_alpha = args.alpha,
                                    min_alpha = args.min_alpha,
                                    alpha_decay = args.alpha_decay,
                                    max_steps_per_episode = args.max_steps_per_episode)
        
            agent.train(env, iters, args.early_stopping_patience_mc)

            # Set the exploration rate to 0 for evaluation
            agent.epsilon = 0
        
        elif agent == "ql":
            print("Using Q-Learning agent")
            
            agent = QLearningAgent(grid_shape = env.grid.shape,
                                   grid_name = grid,
                                   gamma = args.gamma,
                                   initial_epsilon = args.epsilon,
                                   min_epsilon = args.min_epsilon,
                                   epsilon_decay = args.epsilon_decay,
                                   initial_alpha = args.alpha,
                                   min_alpha = args.min_alpha,
                                   alpha_decay = args.alpha_decay,
                                   max_steps_per_episode = args.max_steps_per_episode)


            agent.train(env, iters, args.early_stopping_patience_ql)

            # Set the exploration rate to 0 for evaluation
            agent.epsilon = 0
        
        else:
            raise ValueError(f"Unknown agent type: {agent}")

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


if __name__ == '__main__':
    main_dispatcher()