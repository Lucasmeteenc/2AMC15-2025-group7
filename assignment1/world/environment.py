"""
Gymnasium-compatible Environment.
"""
import random
import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from world.helpers import save_results, action_to_direction
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

try:
    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.append(root_path)

    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path


class Environment(gym.Env):
    """
    A Grid‐world environment compatible with Gymnasium.

    The observation is simply the agent's (row, col) position on the grid.
    The action space is Discrete(4) (0=down,1=up,2=left,3=right).
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self,
                 grid_fp: Path = Path("grid_configs/A1_grid.npy"),
                 render_mode: str = None,
                 sigma: float = 0.,
                 agent_start_pos: tuple[int, int] = None,
                 reward_fn: callable = None,
                 target_fps: int = 30,
                 random_seed: int | float | str | bytes | bytearray | None = 0,
                 max_episode_steps: int = 1000):
        
        """Creates the Grid Environment for the Reinforcement Learning robot
        from the provided file.

        Args:
            grid_fp: Path to the grid file to use.
            render_mode: The render mode ("human").
            sigma: The stochasticity of the environment. The probability that
                the agent makes the move that it has provided as an action is
                calculated as 1-sigma.
            agent_start_pos: Tuple where each agent should start.
                If None is provided, then a random start position is used.
            reward_fn: Custom reward function to use. 
            target_fps: How fast the simulation should run if it is being shown
                in a GUI.
            random_seed: The random seed to use for this environment.
        """
        super().__init__()
        
        # Validate render mode
        self.render_mode = render_mode
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Initialize Grid
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")

        self.grid_fp = grid_fp

        # Load initial grid to determine action and observation spaces
        initial_grid = Grid.load_grid(self.grid_fp).cells
        self.grid_shape = initial_grid.shape
        
        # Determine the number of chargers in the grid - At least one charger is required
        self.nr_chargers = max(1, np.sum(initial_grid == 4))  
        
        # action space: 5 discrete actions (down, up, left, right, charge)
        self.action_space = spaces.Discrete(5)
        
        self.charge = 100.0  # Initial battery charge
        self.max_charge = 100.0  # Maximum battery charge
        self.charger_charge = 20.0 # How much charge a charger gives for one charge action
        self.depletion_rate = 2.0  # Battery charge depletion rate per step
        self.nr_chargers = 40  # Number of chargers in the grid
        
        self.charger_locations = self._generate_charger_positions(grid=initial_grid)  # Ensure at least one charger is present
        
        # Observation space dim
        low = np.array([0, 0] +   # Agent position
                        [0, 0] +  # Target position
                        [0, 0] * self.nr_chargers + # charger positions
                        [0.0], dtype=np.float32) # Battery charge
        
        grid_high = max(self.grid_shape)
        high = np.array([grid_high, grid_high] +
                        [grid_high, grid_high] + 
                        [grid_high, grid_high] * self.nr_chargers + 
                        [self.max_charge], dtype=np.float32)
        
        # New observation space: agent position, target position, charger position, battery charge
        self.observation_space = spaces.Box(
            low=low, 
            high=high, 
            dtype=np.float32
        )
        
        # Initialize other variables
        self.agent_start_pos = agent_start_pos
        self.sigma = sigma
              
        # Set up reward function
        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn

        # GUI specific code
        if target_fps <= 0:
            self.target_spf = 0.
        else:
            self.target_spf = 1. / target_fps
        self.gui = None
        
        # Initialize state variables
        self.grid = None
        self.agent_pos = None
        self.target_pos = None
        self.info = {}
        self.world_stats = {}
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        
    def _generate_charger_positions(self, grid: np.ndarray) -> list[tuple[int, int]]:
        """Generates random charger positions on the grid."""
        charger_positions = []
        zeros = np.where(grid == 0)
        
        while len(charger_positions) < self.nr_chargers:
            idx = random.randint(0, len(zeros[0]) - 1)
            pos = (zeros[0][idx], zeros[1][idx])
            if pos not in charger_positions:
                charger_positions.append(pos)
                grid[pos] = 4
                
        return charger_positions
        
    def _set_chargers_on_grid(self):
        """Set the chargers on the defined grid locations."""
        for pos in self.charger_locations:
            self.grid[pos] = 4


    def _reset_info(self) -> dict:
        """Resets the info dictionary."""
        return {"target_reached": False,
                "agent_moved": False,
                "actual_action": None}
    
    @staticmethod
    def _reset_world_stats() -> dict:
        """Resets the world stats dictionary."""
        return {"cumulative_reward": 0,
                "total_steps": 0,
                "total_agent_moves": 0,
                "total_failed_moves": 0,
                "total_targets_reached": 0,
                }

    def _initialize_agent_pos(self):
        """Initializes agent position from the given location or
        randomly chooses one if None was given.
        """
        if self.agent_start_pos is not None:
            pos = (self.agent_start_pos[0], self.agent_start_pos[1])
            if self.grid[pos] == 0:
                self.agent_pos = pos
            else:
                raise ValueError(
                    "Attempted to place agent on top of obstacle, delivery "
                    "location or charger")
        else:
            warn("No initial agent positions given. Randomly placing agents "
                 "on the grid.")
            zeros = np.where(self.grid == 0)
            idx = random.randint(0, len(zeros[0]) - 1)
            self.agent_pos = (zeros[0][idx], zeros[1][idx])
            
    def _get_state(self) -> np.ndarray:
        """Returns the current state of the environment."""
        agent_pos = np.array(self.agent_pos, dtype=np.float32)
        target_pos = np.array(self.target_pos, dtype=np.float32).flatten()
        charger_positions = np.array(np.where(self.grid == 4)).T.flatten()
        battery_charge = np.array(self.charge, dtype=np.float32)
        
        # Combine all into a single state array
        state = np.concatenate((agent_pos, target_pos, charger_positions, battery_charge.reshape(1)))
        return state.reshape(-1, 1).flatten()  # Flatten to 1D arrays

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (can contain environment parameters).
            
        Returns:
            observation: Initial state.
            info: Additional information dictionary.
        """
        # Handle seeding
        super().reset(seed=seed)
        
        # Handle options
        if options is not None:
            for k, v in options.items():
                match k:
                    case "grid_fp":
                        self.grid_fp = v
                    case "agent_start_pos":
                        self.agent_start_pos = v
                    case "render_mode":
                        self.render_mode = v
                    case "target_fps":
                        self.target_spf = 1. / v if v > 0 else 0.
                    case _:
                        warn(f"{k} is not one of the recognized options.")
        
        # Reset variables
        self.grid = Grid.load_grid(self.grid_fp).cells
        self._initialize_agent_pos()
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()
        self._set_chargers_on_grid()
        self.charge = self.max_charge  # Reset charge to max
        self.target_pos = np.where(self.grid == 3)

        # GUI specific code
        if self.render_mode == "human":
            if self.gui is None:
                self.gui = GUI(self.grid.shape)
            self.gui.reset()
        else:
            if self.gui is not None:
                self.gui.close()
                self.gui = None

        self.current_step = 0
        # return np.array(self.agent_pos, dtype=np.int32), self.info
        return self._get_state(), self.info

    def _move_agent(self, new_pos: tuple[int, int]):
        """Moves the agent, if possible and updates the corresponding stats."""
        match self.grid[new_pos]:
            case 0 | 4:  # Moved to an empty tile
                self.agent_pos = new_pos
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
            case 1 | 2:  # Moved to a wall or obstacle
                self.world_stats["total_failed_moves"] += 1
                self.info["agent_moved"] = False
            case 3:  # Moved to a target tile
                self.agent_pos = new_pos
                self.grid[new_pos] = 0
                self.info["target_reached"] = True
                self.world_stats["total_targets_reached"] += 1
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
            case _:
                raise ValueError(f"Grid is badly formed. It has a value of "
                                f"{self.grid[new_pos]} at position {new_pos}.")

    def step(self, action: int):
        """Execute one time step within the environment.

        Args:
            action: Integer representing the action (0=down, 1=up, 2=left, 3=right).

        Returns:
            observation: Current state after the action.
            reward: Reward for the action.
            terminated: Whether the episode has ended (target reached).
            truncated: Whether the episode was truncated (always False here).
            info: Additional information dictionary.
        """
        self.world_stats["total_steps"] += 1
        self.current_step += 1
        
        # GUI specific code for pausing
        is_single_step = False
        if self.render_mode == "human" and self.gui is not None:
            start_time = time()
            while self.gui.paused:
                if self.gui.step:
                    is_single_step = True
                    self.gui.step = False
                    break
                paused_info = self._reset_info()
                paused_info["agent_moved"] = True
                self.gui.render(self.grid, self.agent_pos, paused_info, 0, is_single_step, self.world_stats)

        # Add stochasticity to the agent action
        val = random.random()
        if val > self.sigma:
            actual_action = action
        else:
            actual_action = random.randint(0, 4)
        
        # Make the move
        self.info["actual_action"] = actual_action
        direction = action_to_direction(actual_action)    
        new_pos = (self.agent_pos[0] + direction[0], self.agent_pos[1] + direction[1])

        # Calculate the reward for the agent moving
        reward = self.reward_fn(self.grid, new_pos)
        self._move_agent(new_pos)
        
        # Check charger penalty
        if actual_action == 4:
            # Charging on a non-charger tile
            if self.grid[self.agent_pos] != 4:
                reward = -5
            else:
                # print("Charging on a charger tile.")
                self.charge += self.charger_charge
                
                # Check overcharge
                if self.charge > self.max_charge:
                    self.charge = self.max_charge
                    reward = -3
                else:
                    reward = -1.0
                    

        # Check if terminal state is reached
        terminated = np.sum(self.grid == 3) == 0
        truncated = self.current_step >= self.max_episode_steps
        
        # Battery depletion
        self.deplete_battery()
        if self.charge <= 0:
            self.charge = -1.0 
            
            # Allow move, but switched to "gasoline" mode: Very expensive moves
            reward -= 10

        self.world_stats["cumulative_reward"] += reward

        # Render if needed
        if self.render_mode == "human" and self.gui is not None:
            time_to_wait = self.target_spf - (time() - start_time)
            if time_to_wait > 0:
                sleep(time_to_wait)
            self.gui.render(self.grid, self.agent_pos, self.info, reward, is_single_step, self.world_stats)
            
        if terminated or truncated:
            print(f"Episode ended after {self.current_step} steps. "
                    f"Total reward: {self.world_stats['cumulative_reward']}")

        # return np.array(self.agent_pos, dtype=np.int32), reward, terminated, truncated, self.info
        return self._get_state(), reward, terminated, truncated, self.info
    
    def deplete_battery(self):
        """Deplete the battery following randomly chosen depletion rate."""
        mean = self.depletion_rate
        sigma = self.depletion_rate / 3.0    # ⇒ ~99.7% of a N(μ,σ²) lies in [μ−3σ, μ+3σ] == [0, 2·depletion_rate]
        
        depletion = random.gauss(mean, sigma)
        
        # Bound depletion to be within [0, 2·depletion_rate]
        depletion = min(max(0.0, depletion), self.depletion_rate * 2)  
        
        self.charge -= depletion
        
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.gui is None:
                self.gui = GUI(self.grid.shape)
            self.gui.render(self.grid, self.agent_pos, self.info, 0, False, self.world_stats)
            
            # Hack to allow baseline
        elif self.render_mode == "rgb_array":
            # 1) Create a fresh figure & axis
            fig, ax = plt.subplots(
                figsize=(self.grid.shape[1] / 10, self.grid.shape[0] / 10),
                dpi=100
            )
            ax.imshow(self.grid, cmap="gray_r", interpolation="nearest")
            ax.scatter(
                self.agent_pos[1],
                self.agent_pos[0],
                c="red",
                s=50,
                marker="o"
            )
            # … any other plotting code you want (targets, chargers, etc.) …
            ax.axis("off")

            # 2) Attach an Agg canvas to that figure and draw
            canvas = FigureCanvas(fig)
            canvas.draw()

            # 3) Extract the RGBA buffer from the Agg canvas
            width, height = canvas.get_width_height()
            buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            # buffer.shape == (height * width * 4,)
            img_rgba = buffer.reshape((height, width, 4))

            # 4) Drop the alpha channel → keep only RGB
            img_rgb = img_rgba[:, :, :3]

            plt.close(fig)
            return img_rgb

        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' not supported.")
        
    def close(self):
        """Close the environment."""
        if self.gui is not None:
            self.gui.close()
            self.gui = None

    @staticmethod
    def _default_reward_function(grid, agent_pos) -> float:
        """Default reward function."""
        match grid[agent_pos]:
            case 0:  # Moved to an empty tile
                reward = -1
            case 1 | 2:  # Moved to a wall or obstacle
                reward = -5
            case 3:  # Moved to a target tile
                reward = 10
            case 4:  # Moved to a charger tile    
                reward = -1
            case _:
                raise ValueError(f"Grid cell should not have value: {grid[agent_pos]} "
                                f"at position {agent_pos}")
        return reward

    @staticmethod
    def evaluate_agent(grid_fp: Path,
                       agent: BaseAgent,
                       max_steps: int,
                       sigma: float = 0.,
                       agent_start_pos: tuple[int, int] = None,
                       random_seed: int | float | str | bytes | bytearray = 0,
                       show_images: bool = False):
        """Evaluates a trained agent's performance."""
        env = Environment(grid_fp=grid_fp,
                            render_mode=None,
                            sigma=sigma,
                            agent_start_pos=agent_start_pos,
                            target_fps=-1,
                            random_seed=random_seed)
        
        state, _ = env.reset()
        initial_grid = np.copy(env.grid)
        agent_path = [tuple(env.agent_pos)]

        for _ in trange(max_steps, desc="Evaluating agent"):
            action = agent.take_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            agent_path.append(tuple(state))

            if terminated or truncated:
                break

        env.world_stats["targets_remaining"] = np.sum(env.grid == 3)
        path_image = visualize_path(initial_grid, agent_path)
        file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        save_results(file_name, env.world_stats, path_image, show_images)
        
        env.close()