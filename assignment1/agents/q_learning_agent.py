import numpy as np

from agents import BaseAgent
from world import Environment
from tqdm import trange


class QLearningAgent(BaseAgent):
    def __init__(self, grid_shape, grid_name: str, num_actions=4, gamma=0.9, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.999, 
                 initial_alpha=1.0, min_alpha=0.01, alpha_decay=0.999, stochasticity=-1, max_steps_per_episode=-1, reward_function="Default"):
        """
        Initialize the Q-learning Agent.

        Args:
            grid_shape (tuple): (height, width) of the grid.
            grid_name (str): Name of the grid file.
            num_actions (int): Number of possible actions.
            gamma (float): Discount factor.
            initial_epsilon (float): Starting value for epsilon (exploration rate).
            min_epsilon (float): Minimum value for epsilon.
            epsilon_decay (float): Factor to decay epsilon by each episode.
            initial_alpha (float): Starting value for alpha (exploration rate).
            min_alpha (float): Minimum value for alpha.
            alpha_decay (float): Factor to decay alpha by each episode.
            stochasticity (float): Stochasticity of the environment.
            max_steps_per_episode (int): Maximum steps allowed per episode.
            reward_function (str): Reward function to use.
        """
        super().__init__()

        self.num_actions = num_actions
        self.grid_height, self.grid_width = grid_shape
        self.gamma = gamma

        # Initialize Q-table
        self.Q_table = np.zeros((self.grid_width, self.grid_height, self.num_actions))

        # Initialize old Q-Table and old V to compute convergence speed
        self.Q_table_old = np.zeros((self.grid_width, self.grid_height, self.num_actions))
        self.V_old = np.max(self.Q_table,axis=2)

        # Epsilon parameters
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = initial_epsilon

        # Alpha parameters
        self.alpha = initial_alpha
        self.min_alpha = min_alpha
        self.alpha_decay = alpha_decay
        self.initial_alpha = initial_alpha

        self.old_state = None
        
        # Early exit if td improvement is too little
        self.little_improvement_steps = 0

        self.max_steps_per_episode = max_steps_per_episode

        self._set_parameters("Q learning", 
                             stochasticity=stochasticity, 
                             discount_factor=gamma, 
                             grid_name=grid_name, 
                             episode_length=max_steps_per_episode, 
                             reward_function=reward_function,
                             initial_alpha=initial_alpha,
                             epsilon_decay=epsilon_decay)

    def update_epsilon(self):
        """
        Update the exploration rate (epsilon) after each episode.
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update_alpha(self):
        """
        Update the learning rate (alpha) after each episode.
        """
        if self.alpha > self.min_alpha:
            self.alpha *= self.alpha_decay

    def update(self, state: tuple[int, int], reward: float, action):
        """
        Update the Q-table based on the action taken and the received reward.

        Args:
            state: The updated position of the agent.
            reward: The value which is returned by the environment as a reward.
            action: The action which was taken by the agent.
        """
        # TD error
        TD_error = reward + self.gamma * np.max(self.Q_table[state[0],state[1]]) - self.Q_table[self.old_state[0], self.old_state[1], action]

        # Q-learning update
        self.Q_table[self.old_state[0], self.old_state[1], action] = self.Q_table[self.old_state[0], self.old_state[1], action] + self.alpha * TD_error
        
        if abs(self.alpha * TD_error) < 1e-6:
            self.little_improvement_steps += 1
        else:
            self.little_improvement_steps = 0

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state (tuple): Current state (row, col).

        Returns:
            int: The chosen action (0-3).
        """
        self.old_state = state

        # Epsilon greedy
        if np.random.random() > self.epsilon:
            # Explore: choose a random action
            return np.argmax(self.Q_table[state[0], state[1]])
        else:
            # Exploit: choose the best action based on current Q-values
            return np.random.randint(self.num_actions)
        
    def train(self, env: Environment, num_episodes: int, early_stopping_patience: int):

        print(f"\n--- Training Q-learning Agent on Grid: {env.grid_fp.name} ---")

        init_grid_for_policy_print = np.copy(env.grid)
        
        # Main training loop
        for episode in trange(num_episodes):
            
            state = env.reset()
            self.update_epsilon()
            self.update_alpha()

            self.Q_table_old = np.copy(self.Q_table)

            for _ in range(self.max_steps_per_episode):
                
                # Agent takes an action based on the latest observation and info.
                action = self.take_action(state)

                # The action is performed in the environment
                state, reward, terminated, info = env.step(action)
                
                # If the final state is reached, stop.
                if terminated:
                    break

                self.update(state, reward, info["actual_action"])

                self.step += 1
                
            conv_metricV,conv_metricQ = self.get_convergence_metric()
            self.log_metrics(env.world_stats["cumulative_reward"], self.alpha, self.epsilon,conv_metricV,conv_metricQ)
            self.episode += 1

            if self.little_improvement_steps > early_stopping_patience:
                print(f"Early exit after {episode} episodes.")
                break

        self.print_policy(init_grid_for_policy_print)

    def get_convergence_metric(self):
        """
        Computes value functions for current and previous Q-tables and returns absolute
        difference between current and previous value function.
        """
        
        # Compute current value function
        V = np.max(self.Q_table, axis=2)
        # Compute max difference between value functions
        max_diff_V = np.max(np.abs(V - self.V_old))
        # Update old value function for next iteration
        self.V_old = V

        # Compute abs difference between Q-tables
        abs_diff = np.abs(self.Q_table - self.Q_table_old)
        # Compute max difference between Q-tables
        max_diff_Q = np.max(abs_diff)

        return max_diff_V, max_diff_Q
    
    def get_policy(self):
        """
        Extract the policy from the Q-table.

        Returns:
            np.ndarray: The policy (best action for each state).
        """
        return np.argmax(self.Q_table, axis=2)
    
    def print_policy(self, init_grid):
        """
        Print the policy in a human-readable format.

        Args:
            init_grid (np.ndarray): The grid environment.
        """
        print("\nPolicy (best action for each state):")
        found_policy = self.get_policy()
        H, W = found_policy.shape

        action_symbols = {
            0: '↓',  # Down
            1: '↑',  # Up
            2: '←',  # Left
            3: '→'   # Right
        }
        wall_symbol = '#'

        WALL_VALUE = 1
        OBSTACLE_VALUE = 2
        TARGET_VALUE = 3

        print("-" * (H * 2 + 1))

        # Print the transposed policy row by row
        for r_vis in range(W):
            row_str = "|"
            for c_vis in range(H):
                original_row = c_vis
                original_col = r_vis

                if init_grid[original_row, original_col] == WALL_VALUE or init_grid[original_row, original_col] == OBSTACLE_VALUE:
                    row_str += wall_symbol + " "
                elif init_grid[original_row, original_col] == TARGET_VALUE:
                    row_str += "T" + " "
                else:
                    action = found_policy[original_row, original_col]
                    row_str += action_symbols.get(action, '?') + " "
            row_str += "|"
            print(row_str)

        print("-" * (H * 2 + 1))