"""vi Agent.

This is an agent that takes the action wich maximizes the value of the value function.
"""
import numpy as np

from world.grid import Grid
from agents import BaseAgent
from world.helpers import action_to_direction


class ViAgent(BaseAgent):
    """Agent that performs Value Iteration to find optimal policy."""
    def __init__(self, grid: Grid, grid_size: tuple[int, int], reward: callable, gamma: float = 0.9, sigma: float = 0.1):
        super().__init__()
        
        self.grid_size = grid_size
        self.V = np.zeros(grid_size)    # Value function
        self.gamma = gamma              # Discount factor
        self.actions = [0, 1, 2, 3]     # Possible actions (Down, Up, Left, Right)
        self.reward_fn = reward         # Reward function
        self.grid = grid                # Grid environment
        self.theta = 10**-6              # Convergence threshold
        self.sigma = sigma              # Stochasticity that agent takes random action
        
        # Initialize the value function
        self.initialize_values()
    
    def initialize_values(self):
        """Initialize the value function based on grid information."""
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i, j] in [1, 2]:  # Wall or obstacle
                    self.V[i, j] = -100.0
                elif self.grid[i, j] == 3:     # Target
                    self.V[i, j] = 10.0
    
    def value_iteration(self):
        """Perform one sweep of value iteration over the state space."""
        
        delta = 0
        
        # For each state
        for i in range(1, self.grid_size[0]-1):
            for j in range(1, self.grid_size[1]-1):
                if self.grid[i, j] in [1, 2]:  # Skip walls and obstacles
                    continue
                    
                # Store old value
                old_v = self.V[i, j]
                
                # Compute the value of each action
                action_values = []
                
                for a in self.actions:
                    # Initialize expected value for this action
                    expected_value = 0
                    
                    # Consider all possible outcomes due to stochasticity
                    # With probability (1-sigma), the chosen action is taken
                    # With probability sigma, a random action is taken
                    
                    # For the intended action (probability 1-sigma)
                    intended_next_state = tuple(np.add((i, j), action_to_direction(a)))
                    intended_reward = self.reward_fn(self.grid, intended_next_state)
                    expected_value += (1 - self.sigma) * (intended_reward + self.gamma * self.V[intended_next_state])
                    
                    # For random actions (probability sigma/(len(actions)) for each)
                    other_action_prob = self.sigma / len(self.actions)
                    
                    for random_a in self.actions:
                        random_next_state = tuple(np.add((i, j), action_to_direction(random_a)))
                        random_reward = self.reward_fn(self.grid, random_next_state)
                        expected_value += other_action_prob * (random_reward + self.gamma * self.V[random_next_state])
                    
                    action_values.append(expected_value)
                
                # Update value with the max of all action values
                self.V[i, j] = max(action_values)
                
                # Track the maximum change
                delta = max(delta, abs(old_v - self.V[i, j]))
        
        return delta
    
    def update(self, state: tuple[int, int], reward: float, action):
        """Since we learn from the model and not from experiences of the environment"""
        pass
    
    def take_action(self, state: tuple[int, int]) -> int:
        """Choose the action that maximizes expected future reward."""
        
        best_action = 0
        best_value = float('-inf')
        
        for a in self.actions:
            next_state = tuple(np.add(state, action_to_direction(a)))
            r = self.reward_fn(self.grid, next_state)
            value = r + self.gamma * self.V[next_state]
            if value > best_value:
                best_value = value
                best_action = a
        
        return best_action
    
    def get_policy(self):
        """Extract the complete policy from the value function for all states."""
        policy = np.zeros(self.grid_size, dtype=int)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                policy[i, j] = self.take_action((i, j)) 
        return policy