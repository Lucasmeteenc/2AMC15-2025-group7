"""Q-learning Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np

from agents import BaseAgent
from world.grid import Grid


class QLearningAgent(BaseAgent):
    def __init__(self, grid: np.ndarray, gamma: float, nr_actions: int = 4):
        """Base agent. All other agents should build on this class.

        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """
        self.nr_actions = nr_actions

        self.n_cols, self.n_rows= grid.shape

        self.Q_table = np.zeros((self.n_rows, self.n_cols, self.nr_actions))
        self.gamma = gamma

        self.alpha = 0.1
        self.epsilon = 1.0

        self.old_state = None

    def decay_learning_params(self, nEpisodes: int, episode: int):
        if episode > 0.3*nEpisodes:
            self.alpha = self.alpha * 0.9995      #initial_alpha / (1 + i / 1000)
            self.epsilon = self.epsilon * 0.999  #max(min_epsilon, initial_epsilon * np.exp(-i / decay_rate))

    """Agent that performs a random action every time. """
    def update(self, state: tuple[int, int], reward: float, action):
        """Any code that processes a reward given the state and updates the agent.

        Args:
            state: The updated position of the agent.
            reward: The value which is returned by the environment as a
                reward.
            action: The action which was taken by the agent.
        """
        pass

        # TD error
        TD_error = reward + self.gamma * np.max(self.Q_table[state[0],state[1]]) - self.Q_table[self.old_state[0], self.old_state[1], action]

        # Q-learning update
        self.Q_table[self.old_state[0], self.old_state[1], action] = self.Q_table[self.old_state[0], self.old_state[1], action] + self.alpha * TD_error


    def take_action(self, state: tuple[int, int], evaluate: bool = False) -> int:
        """Any code that does the action should be included here.

        Args:
            state: The updated position of the agent.
        """
        self.old_state = state

        # Epsilon greedy
        if evaluate or np.random.random() > self.epsilon:
            return np.argmax(self.Q_table[state[0], state[1]])
        else:
            return np.random.randint(self.nr_actions)