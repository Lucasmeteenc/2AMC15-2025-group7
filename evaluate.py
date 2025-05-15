"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import random
import multiprocessing as mp

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
    # p.add_argument("--sigma", type=float, default=0.1,
    #                help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    # p.add_argument("--iters", type=int, default=1000,
    #                help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    # p.add_argument("--agent", type=str, default="vi",
    #                help="Agent selection: vi (Value Iteration), mc (On Policy Monte Carlo), ql (Q-Learning).")
    
    # Monte Carlo specific parameters
    # p.add_argument("--max_steps_per_episode", type=int, default=500, # Safety limit
    #                help="Maximum steps allowed per episode.")
    # p.add_argument("--gamma", type=float, default=0.9, # Discount factor
    #                help="Discount factor gamma.")
    # p.add_argument("--epsilon", type=float, default=1.0, # Initial Epsilon
    #                help="Initial exploration rate epsilon.")
    # p.add_argument("--min_epsilon", type=float, default=0.05, # Minimum Epsilon
    #                help="Minimum exploration rate epsilon.")
    # p.add_argument("--epsilon_decay", type=float, default=0.9995, # Epsilon decay rate
    #                help="Epsilon decay rate per episode.")
    # p.add_argument("--early_stopping_patience_mc", type=int, default=250,
    #                help="Amount of episodes with the same policy that triggers early stopping.")
    
    # Q-Learning specific parameters
    # p.add_argument("--early_stopping_patience_ql", type=int, default=50,
    #                help="Amount of episodes with the same policy that triggers early stopping.")
    return p.parse_args()


def run_train_loop(agent, grid, no_gui, num_episodes, fps, sigma, gamma, epsilon, min_epsilon, epsilon_decay, max_steps_per_episode, early_stopping_patience):
    # Define random seed such that each loop is different.
    random_seed = random.randint(-1000000000, 1000000000)
    
    env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)

    # Always reset the environment to initial state
    _ = env.reset()

    if agent == "vi":
        print("Using Value Iteration agent")

        agent = ViAgent(grid_name=grid, 
                        grid_size=env.grid.shape, 
                        grid=env.grid, 
                        gamma=gamma, 
                        reward=env.reward_fn, 
                        sigma=sigma)

        agent.train(env)

    elif agent == "mc":
        print("Using On Policy Monte Carlo agent")

        agent = MonteCarloAgent(grid_shape=env.grid.shape,
                                grid_name=grid,
                                gamma=gamma,
                                initial_epsilon=epsilon,
                                min_epsilon=min_epsilon,
                                epsilon_decay=epsilon_decay,
                                stochasticity=sigma,
                                max_steps_per_episode=max_steps_per_episode)
    
        agent.train(env, num_episodes, early_stopping_patience)

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

        agent.train(env, num_episodes, max_steps_per_episode, early_stopping_patience)

        # Set the exploration rate to 0 for evaluation
        agent.epsilon = 0
    
    else:
        raise ValueError(f"Unknown agent type: {agent}")
    
    # evaluate (not needed currently)
    # Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


def main_dispatcher():
    args = parse_args()

    no_gui = args.no_gui
    fps = args.fps
    random_seed = args.random_seed

    # sigma = args.sigma
    # agent = args.agent
    # num_episodes = args.iters
    # gamma = args.gamma
    # sigma = args.sigma

    # epsilon = args.epsilon
    # min_epsilon = args.epsilon
    # epsilon_decay = args.epsilon_decay
    # max_steps_per_episode = args.max_steps_per_episode

    # early_stopping_patience_ql = args.early_stopping_patience_ql
    # early_stopping_patience_mc = args.early_stopping_patience_mc

    agents = ["vi", "ql", "mc"]

    # Default values
    default_values = {
        'vi': {
            'sigma': 0.1,
            'gamma': 0.95,
            'epsilon': -1,
            'min_epsilon': -1,
            'epsilon_decay': -1,
            'num_episodes': -1,
            'max_steps_per_episode': -1,
            'early_stopping_patience': -1
        },
        'mc': {
            'sigma': 0.1,
            'gamma': 0.95,
            'epsilon': 1.0,
            'min_epsilon': 0.01,
            'epsilon_decay': 0.9997,
            'num_episodes': 10000,
            'max_steps_per_episode': 500,
            'early_stopping_patience': 1000
        },
        'ql': {
            'sigma': 0.1,
            'gamma': 0.95,
            'epsilon': 1.0,
            'min_epsilon': -1,
            'epsilon_decay': -1,
            'num_episodes': 1000,
            'max_steps_per_episode': 500,
            'early_stopping_patience': 50
        }
    }

    # Define values to test for different parameters
    values_to_test = {
        'sigma': [0.01, 0.05, 0.1, 0.3],
        'gamma': [0.4, 0.8, 0.9, 0.95, 0.99],
        # 'epsilon': [1.0, 0.7, 0.3, 0.1],
        # 'num_episodes': [500, 1000, 5000, 10000],
        # 'max_steps_per_episode': [100, 250, 500, 1000],
        # 'early_stopping_patience': [50, 100, 250, 500, 1000]
    }

    # Number of runs
    num_runs = 10
    
    # Create task list for parallel execution
    all_tasks = []
    
    for _ in range(num_runs):
        for agent in agents:
            for grid in args.GRID:
                # Stochasticity
                for mod_sigma in values_to_test['sigma']:
                    all_tasks.append((
                        agent, grid, no_gui, default_values[agent]['num_episodes'], 
                        fps, mod_sigma, default_values[agent]['gamma'], default_values[agent]['epsilon'], 
                        default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                        default_values[agent]['max_steps_per_episode'], default_values[agent]['early_stopping_patience']
                    ))
                
                # Discounted reward
                for mod_gamma in values_to_test['gamma']:
                    all_tasks.append((
                        agent, grid, no_gui, default_values[agent]['num_episodes'], 
                        fps, default_values[agent]['sigma'], mod_gamma, default_values[agent]['epsilon'], 
                        default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                        default_values[agent]['max_steps_per_episode'], default_values[agent]['early_stopping_patience']
                    ))

                if agent != "vi":
                    # Starting epsilon. Is still decayed using the same epsilon_decay 
                    for mod_epsilon in values_to_test['epsilon']:
                        all_tasks.append((
                            agent, grid, no_gui, default_values[agent]['num_episodes'], 
                            fps, default_values[agent]['sigma'], default_values[agent]['gamma'], mod_epsilon, 
                            default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                            default_values[agent]['max_steps_per_episode'], default_values[agent]['early_stopping_patience']
                        ))
                        
                    # Number of episodes
                    for mod_num_episodes in values_to_test['num_episodes']:
                        all_tasks.append((
                            agent, grid, no_gui, mod_num_episodes, 
                            fps, default_values[agent]['sigma'], default_values[agent]['gamma'], default_values[agent]['epsilon'], 
                            default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                            default_values[agent]['max_steps_per_episode'], default_values[agent]['early_stopping_patience']
                        ))

                    # Episode length (both MC and Q-learning, although less efficient for Q-learning.)
                    for mod_max_steps_per_episode in values_to_test['max_steps_per_episode']:
                        all_tasks.append((
                            agent, grid, no_gui, default_values[agent]['num_episodes'], 
                            fps, default_values[agent]['sigma'], default_values[agent]['gamma'], default_values[agent]['epsilon'], 
                            default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                            mod_max_steps_per_episode, default_values[agent]['early_stopping_patience']
                        ))

                        
                    # Early stopping patience
                    for mod_early_stopping_patience in values_to_test['early_stopping_patience']:
                        all_tasks.append((
                            agent, grid, no_gui, default_values[agent]['num_episodes'], 
                            fps, default_values[agent]['sigma'], default_values[agent]['gamma'], default_values[agent]['epsilon'], 
                            default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                            default_values[agent]['max_steps_per_episode'], mod_early_stopping_patience
                        ))
                    # TODO learning rate.

    # Use CPU count - 1 to avoid overwhelming the system
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Running {len(all_tasks)} experiments using {num_processes} parallel processes")
    
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(run_train_loop, all_tasks)

if __name__ == '__main__':
    mp.freeze_support()
    main_dispatcher()