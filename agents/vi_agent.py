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
        self.theta = 0.01               # Convergence threshold
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
                
                # Find the maximum value among all actions
                action_values = []
                for a in self.actions:
                    next_state = tuple(np.add((i, j), action_to_direction(a)))
                    reward = self.reward_fn(self.grid, next_state)
                    
                    # Calculate the expected value
                    expected_value = (1 - self.sigma) * (reward + self.gamma * self.V[next_state])

                    # Calculate probability for each other action
                    other_action_prob = self.sigma / (len(self.actions) - 1)


                    for other_a in self.actions:
                        if other_a == a:
                            continue
                        alt_state = tuple(np.add((i, j), action_to_direction(other_a)))
                        alt_reward = self.reward_fn(self.grid, alt_state)
                        expected_value += other_action_prob * (alt_reward + self.gamma * self.V[alt_state])
                        
                    action_values.append(expected_value)
                
                # Update value with the max
                self.V[i, j] = max(action_values) if action_values else old_v
                
                # Track the maximum change
                delta = max(delta, abs(old_v - self.V[i, j]))
        
        return delta
    
    def update(self, state: tuple[int, int], reward: float, action):
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