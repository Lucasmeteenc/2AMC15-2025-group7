"""vi Agent.

This is an agent that takes the action wich maximizes the value of the value function.
"""
import numpy as np

from world.grid import Grid
from agents import BaseAgent
from world.helpers import action_to_direction
from tqdm import trange
from world.environment import Environment

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
        
        # Initialize the value function see function description for why not used
        # self.initialize_values()
    
    def initialize_values(self):
        """Initialize the value function based on grid information."""
        """Using this gave a worse result than not using it mainly due to the -100 for the walls and the random moving into it
         which is already penalized with the reward"""
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i, j] in [1, 2]:  # Wall or obstacle
                    self.V[i, j] = -100
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
                    if self.grid[i,j] == 3:
                        intended_next_state = (i,j)
                    else:
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


    def print_policy(self, init_grid):
        """
        Print the policy in a human-readable format.

        Args:
            init_grid (np.ndarray): The grid environment.
        """
        print("\nPolicy (best action for each state):")
        found_policy = self.V
        H, W = found_policy.shape

        # action_symbols = {
        #     0: '↓',  # Down
        #     1: '↑',  # Up
        #     2: '←',  # Left
        #     3: '→'  # Right
        # }
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

                if init_grid[original_row, original_col] == WALL_VALUE or init_grid[
                    original_row, original_col] == OBSTACLE_VALUE:
                    row_str += wall_symbol + " "
                elif init_grid[original_row, original_col] == TARGET_VALUE:
                    row_str += "T" + " "
                else:
                    action = found_policy[original_row, original_col]
                    highest_value = -100
                    next_action = ''
                    if found_policy[original_row, original_col+1] > highest_value and init_grid[original_row, original_col+1] != WALL_VALUE:
                        highest_value = found_policy[original_row, original_col+1]
                        next_action = '↓'
                    if found_policy[original_row+1, original_col] > highest_value and init_grid[original_row+1, original_col] != WALL_VALUE:
                        highest_value = found_policy[original_row+1, original_col]
                        next_action = '→'
                    if found_policy[original_row, original_col-1] > highest_value and init_grid[original_row, original_col-1] != WALL_VALUE:
                        highest_value = found_policy[original_row, original_col-1]
                        next_action = '↑'
                    if found_policy[original_row-1, original_col] > highest_value and init_grid[original_row-1, original_col] != WALL_VALUE:
                        highest_value = found_policy[original_row+1, original_col]
                        next_action = '←'
                    row_str += "{} ".format(next_action,1)
            row_str += "|"
            print(row_str)

        # print(found_policy)
        print("-" * (H * 2 + 1))
    
    def train(self, env: Environment):
        delta = float('inf')
        max_iterations = 1000
        
        for iteration in trange(max_iterations):
            # Run one sweep of value iteration
            delta = self.value_iteration()
            
            # Check for convergence
            if delta < self.theta:
                break
                
            # Safety check
            if iteration >= max_iterations - 1:
                print("Warning: Value Iteration did not converge within maximum iterations")
        
        self.print_policy(np.copy(env.grid))