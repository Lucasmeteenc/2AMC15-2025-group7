import numpy as np
import random
from agents import BaseAgent
from world import Environment
from tqdm import trange

class MonteCarloAgent(BaseAgent):
    def __init__(self, grid_shape, grid_name: str, num_actions=4, gamma=0.9, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.999, 
                 initial_alpha=1.0, min_alpha=0.01, alpha_decay=0.999, stochasticity=-1, max_steps_per_episode=-1, reward_function="Default"):
        """
        Initialize the Monte Carlo Agent.

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
        
        self.grid_height, self.grid_width = grid_shape
        self.num_actions = num_actions
        self.gamma = gamma

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

        # Initialize Q-table
        self.Q = np.zeros((self.grid_height, self.grid_width, self.num_actions), dtype=np.float32)

        # Initialize old Q-Table and old V to compute convergence speed
        self.Q_old = np.zeros((self.grid_height, self.grid_width, self.num_actions), dtype=np.float32)
        self.V_old = np.max(self.Q, axis=2)

        self.episode_experience = []
        self.max_steps_per_episode = max_steps_per_episode
        
        self._set_parameters("Monte Carlo", 
                             stochasticity=stochasticity, 
                             discount_factor=gamma, 
                             grid_name=grid_name, 
                             episode_length_mc=max_steps_per_episode, 
                             reward_function=reward_function, 
                             initial_alpha=initial_alpha,
                             epsilon_decay=epsilon_decay)

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state (tuple): Current state (row, col).

        Returns:
            int: The chosen action (0-3).
        """
        row, col = state
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: choose the best action based on current Q-values
            return np.argmax(self.Q[row, col, :])

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

    def update(self, state, reward, action):
        """
        Use the update call (after each step) to store the experience of the current step in the episode.
        
        Args:
            state (tuple): Current state (row, col).
            reward (float): Reward received.
            action (int): Action taken (0-3).
        """

        self.episode_experience.append((state, reward, action))
        
        self.step += 1

    def update_q_from_episode(self):
        """
        Update Q-values and visit counts based on a completed episode.

        Args:
            episode_experience (list): A list of (state, reward, action) tuples for the episode.
        """

        episode_experience = self.episode_experience

        visited = set()

        G = 0
        # Loop backward through the episode
        for t in range(len(episode_experience) - 1, -1, -1):
            state, reward, action = episode_experience[t]
            row, col = state

            # Update return
            G = self.gamma * G + reward

            # First visit implementation
            if (state, action) not in visited:
                # Update Q-value
                current_q = self.Q[row, col, action]
                self.Q[row, col, action] = current_q + self.alpha * (G - current_q)

                visited.add((state, action))

    def train(self, env: Environment, num_episodes: int, early_stopping_patience: int):
        
        print(f"\n--- Training Monte Carlo Agent on Grid: {env.grid_fp.name} ---")

        # Reset variables as safety measure
        self.Q.fill(0)
        self.epsilon = self.initial_epsilon
        self.alpha = self.initial_alpha

        init_grid_for_policy_print = np.copy(env.grid)

        # Initialize variables for early stopping
        old_policy = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        same_policy_count = 0

        converged = False
        
        # Main training loop 
        for episode in trange(num_episodes, desc=f"MC Training on {env.grid_fp.name}"):
            state = env.reset()
            terminated = False
            steps_in_episode = 0
            self.episode_experience = []

            # Store current Q-table for convergence metric
            self.Q_old = np.copy(self.Q)

            # Generate an episode of experience
            while not terminated and steps_in_episode < self.max_steps_per_episode:
                # Agent takes an action based on the current state and policy
                action = self.take_action(state)

                # The action is performed in the environment
                next_state, reward, terminated, _ = env.step(action)

                # Store the experience for this step
                self.update(state, reward, action)

                # Move to the next state
                state = next_state
                steps_in_episode += 1

            # Once the episode is finished update Q-values
            self.update_q_from_episode()

            # Update the exploration rate (epsilon)
            self.update_epsilon()

            # Update the learning rate (alpha)
            self.update_alpha()

            new_policy = self.get_policy()
            if np.array_equal(new_policy, old_policy):
                same_policy_count += 1
                if same_policy_count >= early_stopping_patience:
                    converged = True
                    break
            else:
                same_policy_count = 0
            old_policy = new_policy.copy()
            
            conv_metricV, conv_metricQ = self.get_convergence_metric()
            self.log_metrics(env.world_stats["cumulative_reward"], self.alpha, self.epsilon, conv_metricV, conv_metricQ)
            self.episode += 1

        if converged:
            print(f"\nMC Policy converged after {episode + 1} episodes.")
        
        print("\n--- MC Training completed ---")
        self.print_policy(init_grid_for_policy_print)

    def get_convergence_metric(self):
        """
        Computes value functions for current and previous Q-tables and returns absolute
        difference between current and previous value function.
        """
        
        # Compute current value function
        V = np.max(self.Q, axis=2)
        # Compute max difference between value functions
        max_diff_V = np.max(np.abs(V - self.V_old))
        # Update old value function for next iteration
        self.V_old = V

        # Compute abs difference between Q-tables
        abs_diff = np.abs(self.Q - self.Q_old)
        # Compute max difference between Q-tables
        max_diff_Q = np.max(abs_diff) 

        return max_diff_V, max_diff_Q
    
    def get_policy(self):
        """
        Extract the policy from the Q-table.

        Returns:
            np.ndarray: The policy (best action for each state).
        """
        return np.argmax(self.Q, axis=2)
    
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