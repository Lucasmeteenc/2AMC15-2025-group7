"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
import multiprocessing as mp

try:
    from world import Environment
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
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")

    return p.parse_args()


def run_train_loop(agent, grid, no_gui, num_episodes, fps, sigma, gamma, epsilon, min_epsilon, epsilon_decay, max_steps_per_episode, early_stopping_patience_mc, early_stopping_patience_ql, alpha, min_alpha, alpha_decay, run_id):
    # Define random seed such that each loop is different.
    random_seed = hash(f"{agent}_{grid}_{sigma}_{gamma}_{run_id}") % 1000000

    
    env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)

    # Always reset the environment to initial state
    _ = env.reset()

    if agent == "vi":
        print("Using Value Iteration agent")

        agent = ViAgent(gamma=gamma, grid_size=env.grid.shape, reward=env.reward_fn, grid=env.grid, sigma=sigma)

        agent.train(env)


    elif agent == "mc":
        print("Using On Policy Monte Carlo agent")

        agent = MonteCarloAgent(grid_shape = env.grid.shape,
                                    grid_name = grid,
                                    gamma = gamma,
                                    initial_epsilon = epsilon,
                                    min_epsilon = min_epsilon,
                                    epsilon_decay = epsilon_decay,
                                    initial_alpha = alpha,
                                    min_alpha = min_alpha,
                                    alpha_decay = alpha_decay,
                                    max_steps_per_episode = max_steps_per_episode)
        
        agent.train(env, num_episodes, early_stopping_patience_mc)

        # Set the exploration rate to 0 for evaluation
        agent.epsilon = 0
    
    elif agent == "ql":
        print("Using Q-Learning agent")

        agent = QLearningAgent(grid_shape = env.grid.shape,
                                   grid_name = grid,
                                   gamma = gamma,
                                   initial_epsilon = epsilon,
                                   min_epsilon = min_epsilon,
                                   epsilon_decay = epsilon_decay,
                                   initial_alpha = alpha,
                                   min_alpha = min_alpha,
                                   alpha_decay = alpha_decay,
                                   max_steps_per_episode = max_steps_per_episode)

        agent.train(env, num_episodes, early_stopping_patience_ql)

        # Set the exploration rate to 0 for evaluation
        agent.epsilon = 0
    
    else:
        raise ValueError(f"Unknown agent type: {agent}")


def main_dispatcher():
    args = parse_args()

    no_gui = args.no_gui
    fps = args.fps

    agents = ["vi", "ql", "mc"]

    # Default values
    default_values = {
        'vi': {
            'sigma': 0.1,
            'gamma': 0.99,
            'epsilon': -1,
            'min_epsilon': -1,
            'epsilon_decay': -1,
            'num_episodes': -1,
            'max_steps_per_episode': -1,
            'early_stopping_patience': -1,
            'alpha': -1,
            'min_alpha': -1,
            'alpha_decay': -1

        },
        'mc': {
            'sigma': 0.1,
            'gamma': 0.99,
            'epsilon': 1.0,
            'min_epsilon': 0.0005,
            'epsilon_decay': 0.9999,
            'num_episodes': 100_000,
            'max_steps_per_episode': 5000,
            'early_stopping_patience': 1000,
            'alpha': 0.1,
            'min_alpha': 0.00005,
            'alpha_decay': 0.9999
        },
        'ql': {
            'sigma': 0.1,
            'gamma': 0.99,
            'epsilon': 1.0,
            'min_epsilon': 0.0005,
            'epsilon_decay': 0.9999,
            'num_episodes': 100_000,
            'max_steps_per_episode': 5000,
            'early_stopping_patience': 50,
            'alpha': 0.1,
            'min_alpha': 0.00005,
            'alpha_decay': 0.9999
        },
    }

    # Define values to test for different parameters
    values_to_test = {
        'sigma': [0.0, 0.05, 0.1, 0.3],
        'gamma': [0.4, 0.8, 0.9, 0.95, 0.99],
        'initial_alpha': [0.01, 0.05, 0.1, 0.5, 1],
        'epsilon_decay': [0.95, 0.975, 0.99, 0.999, 0.9999],
        'max_steps_per_episode': [100, 250, 500, 1000],
    }

    # Number of runs
    num_runs = 10
    
    # Create task list for parallel execution
    all_tasks = []
    
    for run_id in range(num_runs):
        for agent in agents:
            for grid in args.GRID:
                # Stochasticity
                for mod_sigma in values_to_test['sigma']:
                    all_tasks.append((
                        agent, grid, no_gui, default_values[agent]['num_episodes'], 
                        fps, mod_sigma, default_values[agent]['gamma'], default_values[agent]['epsilon'], 
                        default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                        default_values[agent]['max_steps_per_episode'], default_values['mc']['early_stopping_patience'],
                        default_values['ql']['early_stopping_patience'], default_values[agent]['alpha'],
                        default_values[agent]['min_alpha'], default_values[agent]['alpha_decay'], run_id
                    ))
                
                # Discounted reward
                for mod_gamma in values_to_test['gamma']:
                    all_tasks.append((
                        agent, grid, no_gui, default_values[agent]['num_episodes'], 
                        fps, default_values[agent]['sigma'], mod_gamma, default_values[agent]['epsilon'], 
                        default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                        default_values[agent]['max_steps_per_episode'], default_values['mc']['early_stopping_patience'],
                        default_values['ql']['early_stopping_patience'], default_values[agent]['alpha'],
                        default_values[agent]['min_alpha'], default_values[agent]['alpha_decay'], run_id
                    ))


                if agent != "vi":
                    # Episode length (both MC and Q-learning, although less efficient for Q-learning.)
                    for mod_max_steps_per_episode in values_to_test['max_steps_per_episode']:
                        all_tasks.append((
                            agent, grid, no_gui, default_values[agent]['num_episodes'], 
                            fps, default_values[agent]['sigma'], default_values[agent]['gamma'], default_values[agent]['epsilon'], 
                            default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                            mod_max_steps_per_episode, default_values['mc']['early_stopping_patience'],
                            default_values['ql']['early_stopping_patience'], default_values[agent]['alpha'],
                            default_values[agent]['min_alpha'], default_values[agent]['alpha_decay'], run_id
                        ))

                    # Initial learning rate
                    for mod_initial_alpha in values_to_test['initial_alpha']:
                        all_tasks.append((
                            agent, grid, no_gui, default_values[agent]['num_episodes'], 
                            fps, default_values[agent]['sigma'], default_values[agent]['gamma'], default_values[agent]['epsilon_decay'], 
                            default_values[agent]['min_epsilon'], default_values[agent]['epsilon_decay'], 
                            default_values[agent]['max_steps_per_episode'], default_values['mc']['early_stopping_patience'],
                            default_values['ql']['early_stopping_patience'], mod_initial_alpha,
                            default_values[agent]['min_alpha'], default_values[agent]['alpha_decay'], run_id
                        ))

                    # Epsilon decay
                    for mod_epsilon_decay in values_to_test['epsilon_decay']:
                        all_tasks.append((
                            agent, grid, no_gui, default_values[agent]['num_episodes'], 
                            fps, default_values[agent]['sigma'], default_values[agent]['gamma'], default_values[agent]['epsilon_decay'], 
                            default_values[agent]['min_epsilon'], mod_epsilon_decay, 
                            default_values[agent]['max_steps_per_episode'], default_values['mc']['early_stopping_patience'],
                            default_values['ql']['early_stopping_patience'], default_values[agent]['alpha'],
                            default_values[agent]['min_alpha'], default_values[agent]['alpha_decay'], run_id
                        ))

    # Use CPU count - 1 to avoid overwhelming the system
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Running {len(all_tasks)} experiments using {num_processes} parallel processes")
    
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(run_train_loop, all_tasks)

if __name__ == '__main__':
    mp.freeze_support()
    main_dispatcher()