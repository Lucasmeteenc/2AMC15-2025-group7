import gymnasium as gym
from gymnasium import spaces
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from maps import MAIL_DELIVERY_MAPS

# Environment parameters
SCALE = 4

MOVE_SIZE = 0.5 * SCALE         # distance covered for forward action
TURN_SIZE = np.deg2rad(15)      # 15 degrees in radians

NOISE_SIGMA = 0.01              # Gaussian noise on x,y after each move
SENSE_RADIUS = 0.5 * SCALE      # radius to check whether pickup-/delivery-

# Simulation parameters
MAX_STEPS = 5_000
NR_PACKAGES = 1

# Rewards
REW_STEP        = -0.2          # per time‐step
REW_PICKUP      = +200.0        # successful pickup
REW_DELIVER     = +250.0        # successful delivery
REW_OBSTACLE    = -0.5          # penalty on hitting an obstacle
REW_WALL        = -0.5          # penalty for going out of bounds

FPS = 30

ACTIONS = {
    0: "Forward",
    1: "Turn left",
    2: "Turn right",
}

class SimpleDeliveryEnv(gym.Env):
    """
    A Gymnasium env for a delivery robot that:
       - picks up packages at a single depot
       - delivers them, one at a time, to random goals
       - must avoid rectangular obstacles
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, map_config: dict = MAIL_DELIVERY_MAPS["default"], render_mode=None, seed=None):
        super().__init__()

        # 1. Init state params
        self.agent_x:            float | None = None
        self.agent_y:            float | None = None
        self.agent_theta:        float | None = None
        self.has_package:         bool | None = None
        self.packages_left:        int | None = None
        self.delivery_goal_x:     float | None = None
        self.delivery_goal_y:     float | None = None

        # 2. Load map data
        self.map_config = self._load_map(map_config)
        self.W,self.H = self.map_size

        # 3. Compute map diagonal for normalization
        self.dmax = np.hypot(self.W, self.H)

        # 4. Episode params
        self.nr_packages = NR_PACKAGES
        self.max_steps = MAX_STEPS

        # 5. Create action space
        self.action_names = ACTIONS
        self.action_space = spaces.Discrete(len(self.action_names))

        # 6. Create (normalized) observation space
        #   1. agent_x / self.map_size[0]   6. packages_left / self.nr_packages
        #   2. agent_y / self.map_size[1]   7. depot_x / self.map_size[0]
        #   3. sin(agent_theta)             8. depot_y / self.map_size[1]
        #   4. cos(agent_theta)             9. delivery_goal_x / self.map_size[0]
        #   5. has_package                  10. delivery_goal_y / self.map_size[1]
        #                                   

        low  = np.array([0, 0, -1, -1, 0, 0, 0, 0, 0, 0],  dtype=np.float32)
        high = np.ones(10, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 7. RNG
        self.rng = np.random.default_rng(seed)

        # 8. flags
        self.hit_wall:          bool | None = None
        self.bumped_obstacle:   bool | None = None

        # 9. Render mode
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        
        # 10. Set one fixed goal
        # self._sample_goal()

    def _load_map(self, map_config: dict):
        """Check whether the provided map_config is a valid map."""
        
        # check whether indeed a dict
        if not isinstance(map_config,dict):
            raise ValueError("map_config must be a dictionary")
        
        # check whether it contains the required keys
        for key in ["size","depot","obstacles", "delivery"]:
            if key not in map_config:
                raise ValueError(f"map_config must contain the key '{key}'")

        # validate and store map size
        sz = map_config["size"]
        if (not isinstance(sz, (tuple, list)) or len(sz) != 2
            or sz[0] <= 0 or sz[1] <= 0):
            raise ValueError("map_config['size'] must be a 2-tuple of positive numbers")
        self.map_size = np.array(sz, dtype=np.float32)

        # validate & store "depot" (must be 2 floats/ints)
        depot = map_config["depot"]
        if (not isinstance(depot, (tuple, list)) or len(depot) != 2):
            raise ValueError("map_config['depot'] must be a 2-tuple (x, y)")
        self.depot = np.array(depot, dtype=np.float32)

        # validate & store "obstacles" (list of 4-tuples)
        obstacles = map_config["obstacles"]
        if (not isinstance(obstacles, (list, tuple)) or
            any((not isinstance(o, (tuple, list)) or len(o) != 4) for o in obstacles)):
            raise ValueError("map_config['obstacles'] must be a list of 4-tuples [(xmin,ymin,xmax,ymax), …]")
        self.obstacles = np.array(obstacles, dtype=np.float32)
        
        # Validate & store "delivery" (must be 2 floats/ints)
        delivery = map_config.get("delivery", None)
        if delivery is None:
            self._sample_goal()
        else:
            if (not isinstance(delivery, (tuple, list)) or len(delivery) != 2):
                raise ValueError("map_config['delivery'] must be a 2-tuple (x, y)")
            self.delivery_goal_x, self.delivery_goal_y = delivery

        return map_config
    
    def reset(self,*,seed=None, options=None):
        """
        Reset the environment to an initial state to start a new episode.
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for resetting the environment.
        Returns:
            observation (np.ndarray): Initial observation of the environment.
            info (dict): Additional information about the reset state.
        """

        # options is a standard gym parameter in reset, but we dont use it
        if options:
            print("Warning: options supplied but not used:", options)

        # set random seed
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # reset agent
        self.agent_x, self.agent_y = self._sample_free_position()
        self.agent_theta = self.rng.uniform(-np.pi,np.pi)

        # reset environmental params
        self.has_package = False
        self.packages_left = self.nr_packages
        
        self.steps = 0

        # initialize flags
        self.hit_wall = False
        self.bumped_obstacle = False

        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int):
        "Advance the simulation by one step given an action."

        # check if env has had its initial reset 
        if self.agent_x is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")
        
        #! reset all flags (not needed bc reset in specific action methods, but kept in, in case we do need it?)
        # self.hit_wall = False
        # self.bumped_obstacle = False

        # execute action
        match action:
            case 0: self._move(MOVE_SIZE)
            case 1: self._turn(+TURN_SIZE)
            case 2: self._turn(-TURN_SIZE)
            case _:
                raise ValueError(f"Invalid action: '{action}'.")

        reward = self._reward_fn(action)
        terminated = self.packages_left == 0
        self.steps+=1
        truncated = (self.steps >= self.max_steps)
        
        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        "Return a 10-dimensional observation vector."

        # relative vectors to static landmarks
        vec_depot   = (self.depot - [self.agent_x, self.agent_y]) / self.dmax

        # relative vectors to current goal if package is held
        if self.has_package:
            vec_goal = (np.array([self.delivery_goal_x, self.delivery_goal_y]) -
                        np.array([self.agent_x, self.agent_y])) / self.dmax
        else:
            vec_goal = np.zeros(2, dtype=np.float32)

        return np.array([
            self.agent_x / self.map_size[0],
            self.agent_y / self.map_size[1],
            np.sin(self.agent_theta),
            np.cos(self.agent_theta),
            float(self.has_package),
            self.packages_left / self.nr_packages,
            vec_depot[0],  vec_depot[1],
            vec_goal[0],   vec_goal[1],
        ], dtype=np.float32)

    # --- Motion Action Checks --------------------------------------

    def _move(self,dist):
        prev_x,prev_y = self.agent_x, self.agent_y
        self.agent_x += dist * np.cos(self.agent_theta)
        self.agent_y += dist * np.sin(self.agent_theta)
        self._post_motion(prev_x, prev_y)

    def _turn(self, dtheta):
        prev_x, prev_y    = self.agent_x, self.agent_y
        self.agent_theta += dtheta
        self._post_motion(prev_x, prev_y)

    def _post_motion(self, prev_x, prev_y):
        """Apply noise, check for collisions or out of bound movement and update the state."""
        self.bumped_obstacle = False
        self.hit_wall        = False

        # apply noise
        self.agent_x += self.rng.normal(0, NOISE_SIGMA)
        self.agent_y += self.rng.normal(0, NOISE_SIGMA)

        # revert to previous position if out of bounds
        if not (
            0 <= self.agent_x <= self.map_size[0]
            and 0 <= self.agent_y <= self.map_size[1]):
            self.agent_x, self.agent_y = prev_x, prev_y
            self.hit_wall = True

        # revert to previous position if colliding with obstacles
        if self._in_obstacle(self.agent_x, self.agent_y):
            self.agent_x, self.agent_y = prev_x, prev_y
            self.bumped_obstacle = True        

    # --- Other Action Checks ---------------------------------------

    def _legal_pickup(self):
        return (
            (not self.has_package)
            and (self.packages_left > 0)
            and self._near(self.depot)
        )

    def _try_pickup(self) -> bool:
        """Attempt Pick-Up
        Returns: True iff it succeeded."""

        if self._legal_pickup():
            self.has_package = True
            return True
        return False

    def _legal_deliver(self):
        return (
            self.has_package 
            and self._near([self.delivery_goal_x, self.delivery_goal_y])
        )

    def _try_deliver(self):
        """Attempt Deliver; returns True iff it succeeded."""

        if self._legal_deliver():
            self.has_package = False
            self.packages_left -= 1
            return True
        return False
    
    # --- Random Delivery Goal Sampling -----------------------------
    def _sample_goal(self):
        """Choose a customer address with 0.7 m clearance from depot """
        while True:
            gx = self.rng.uniform(0, self.map_size[0])
            gy = self.rng.uniform(0, self.map_size[1])
            if (np.hypot(gx - self.depot[0], gy - self.depot[1]) > 0.7):
                # check whether inside an obstacle
                if self._in_obstacle(gx, gy):
                    continue
                # set goal
                self.delivery_goal_x, self.delivery_goal_y = gx, gy
                break
    
    # --- Random Free Position Sampling -----------------------------
    def _sample_free_position(self):
        """
        Returns a random (x,y) uniformly on [0,W]x[0,H] that does NOT lie in any obstacle.
        """
        while True:
            x = self.rng.uniform(0, self.W)
            y = self.rng.uniform(0, self.H)
            if not self._in_obstacle(x, y):
                return x, y

    # --- Reward Function -------------------------------------------
    def _reward_fn(self, action: int):
        # time penalty
        r = REW_STEP

        # action outcomes
        match action:
            # forward, turn left, turn right
            case 0 | 1 | 2:
                # collision & boundary penalties
                if self.bumped_obstacle:           # set in _post_motion()
                    r += REW_OBSTACLE
                if self.hit_wall:                  # set in _post_motion()
                    r += REW_WALL
                # if action in (1,2):
                #     r-= 3 # turning is more costly than moving forward
                    
        # Check if picking up or delivering a package
        if self._try_pickup():
            r += REW_PICKUP
        elif self._try_deliver():
            r += REW_DELIVER

        return r
    
    # --- Geometry Utilities ----------------------------------------
    def _near(self, points, radius=SENSE_RADIUS):
        """Checks whether the agent is within SENSE_RADIUS of some points."""
        pts = np.atleast_2d(points)
        return np.any(np.linalg.norm(pts - [self.agent_x, self.agent_y], axis=1) <= radius)

    def _in_obstacle(self, x, y):
        """Checks whether the agent collides with any obstacle."""
        if self.obstacles.size == 0:
            return False
        xmin, ymin, xmax, ymax = self.obstacles.T
        return np.any((xmin <= x) & (x <= xmax) & (ymin <= y) & (y <= ymax))
    
    # --- 3.11  render & close --------------------------------------
    
    def render(self):
        """
        If render_mode == 'human': show a Matplotlib window that updates in real time.
        If render_mode == 'rgb_array': return a H×W×3 uint8 array of the current frame.
        """
        if self.render_mode not in ("human", "rgb_array"):
            return None

        # 1. Create figure/axis on first call
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6,6))

        # 2. Clear and set up plot limits
        self.ax.clear()
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("SimpleDeliveryEnv")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # 3. Draw obstacles (gray rectangles)
        for obs in self.obstacles:
            rect = Rectangle((obs[0], obs[1]), obs[2]-obs[0], obs[3]-obs[1],
                            color="gray", alpha=0.5)
            self.ax.add_patch(rect)

        # 4. Draw depot as a small green square
        self.ax.add_patch(
            Rectangle((self.depot[0]-0.2, self.depot[1]-0.2), 0.4 * SCALE, 0.4 * SCALE, color="green")
        )

        # 6. If carrying, draw current goal as a red “X”
        if self.has_package:
            gx, gy = self.delivery_goal_x, self.delivery_goal_y
            offset = 0.2 * SCALE
            self.ax.plot([gx-offset, gx+offset], [gy-offset, gy+offset], color="red", linewidth=2 * SCALE)
            self.ax.plot([gx-offset, gx+offset], [gy+offset, gy-offset], color="red", linewidth=2 * SCALE)

        # 7. Draw robot as a blue circle + arrow for orientation
        robot_circle = Circle((self.agent_x, self.agent_y), 0.3, color="blue", alpha=0.8)
        self.ax.add_patch(robot_circle)
        dx = 0.5 * np.cos(self.agent_theta)
        dy = 0.5 * np.sin(self.agent_theta)
        self.ax.arrow(self.agent_x, self.agent_y, dx, dy,
                    head_width=0.15, head_length=0.15, fc="blue", ec="blue")

        # 8. Draw package count in top-left
        pkg_text = "Carrying" if self.has_package else "Empty"
        self.ax.text(0.02 * self.W, 0.94 * self.H,
                    f"{pkg_text}, Left: {self.packages_left}", color="black",
                    fontsize=8, verticalalignment="top")

        # 9. Finalize for human or rgb_array
        if self.render_mode == "human":
            plt.pause(0.001)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            # Grab RGBA buffer from the figure
            buf, (width, height) = self.fig.canvas.print_to_buffer()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
            return img[:, :, :3]   # drop alpha channel, return H×W×3 uint8
    
    def close(self):
        pass
