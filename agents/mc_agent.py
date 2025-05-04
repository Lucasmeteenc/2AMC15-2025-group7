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
        """Decay epsilon."""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update(self, state, reward, action):
        self.episode_experience.append((state, reward, action))

    # Helper function for the main training loop
    def update_q_from_episode(self):
        """
        Update Q-values and visit counts based on a completed episode.
        This should be called from the main training loop in train.py.

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
            
            alpha = 1.0 / self.N_visits[row, col, action]
            # alpha = (1 / (1 + self.N_visits[row, col, action]))**0.5

            # Update Q-value
            current_q = self.Q[row, col, action]
            self.Q[row, col, action] = current_q + alpha * (G - current_q)

    # Helper function to extract the policy
    def get_policy(self):
        """
        Extract the policy from the Q-table.

        Returns:
            np.ndarray: The policy (best action for each state).
        """
        return np.argmax(self.Q, axis=2)