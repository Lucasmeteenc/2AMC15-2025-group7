import numpy as np
import random
from agents import BaseAgent

class MonteCarloAgent(BaseAgent):
    def __init__(self, grid_shape, num_actions=4, gamma=0.9, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.999):
        """
        Initialize the Monte Carlo Agent.

        Args:
            grid_shape (tuple): (height, width) of the grid.
            num_actions (int): Number of possible actions.
            gamma (float): Discount factor.
            initial_epsilon (float): Starting value for epsilon (exploration rate).
            min_epsilon (float): Minimum value for epsilon.
            epsilon_decay (float): Factor to decay epsilon by each episode.
        """
        self.grid_height, self.grid_width = grid_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table and Visit Counts
        self.Q = np.zeros((self.grid_height, self.grid_width, self.num_actions), dtype=np.float32)
        self.N_visits = np.zeros((self.grid_height, self.grid_width, self.num_actions), dtype=np.uint32)

        self.episode_experience = [] 

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

    def update(self, state, reward, action):
        """
        Use the update call (after each step) to store the experience of the current step in the episode.
        
        Args:
            state (tuple): Current state (row, col).
            reward (float): Reward received.
            action (int): Action taken (0-3).
        """

        self.episode_experience.append((state, reward, action))

    def update_q_from_episode(self):
        """
        Update Q-values and visit counts based on a completed episode.

        Args:
            episode_experience (list): A list of (state, reward, action) tuples for the episode.
        """

        episode_experience = self.episode_experience

        G = 0
        # Loop backward through the episode
        for t in range(len(episode_experience) - 1, -1, -1):
            state, reward, action = episode_experience[t]
            row, col = state

            # Update return
            G = self.gamma * G + reward

            # Every Visit MC Update
            self.N_visits[row, col, action] += 1
            
            alpha = 1.0 / (self.N_visits[row, col, action]**0.5)
            
            # Update Q-value
            current_q = self.Q[row, col, action]
            self.Q[row, col, action] = current_q + alpha * (G - current_q)

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