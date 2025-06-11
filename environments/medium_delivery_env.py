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
SENSE_RADIUS = 0.5 * SCALE      # radius to check whether pickup-/delivery-point is in range

# Region rays
NUM_REGIONS = 8                             # number of regions around the agent
REGION_FOV = np.deg2rad(360/NUM_REGIONS)    # fov for each region
MAX_LIDAR_DISTANCE = 2.0                    # max distance for each lidar ray region

# Simulation parameters
MAX_STEPS = 1_000
NR_PACKAGES = 1

# Rewards
REW_PICKUP      = +200.0        # successful pickup
REW_DELIVER     = +250.0        # successful delivery

# Penalties
REW_STEP        = -0.2          # per time‐step
REW_OBSTACLE    = -5            # penalty on hitting an obstacle
REW_WALL        = -5            # penalty for going out of bounds

FPS = 30

ACTIONS = {
    0: "Forward",
    1: "Turn left",
    2: "Turn right",
}

class MediumDeliveryEnv(gym.Env):
    """
    A Gymnasium env for a delivery robot that:
       - picks up packages at a single depot
       - delivers them, one at a time, to random goals
       - must avoid rectangular obstacles
       - has a lidar sensor that provides distance readings
       - no battery recharging, and only moving forward or turning
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
        #   1. agent_x / self.map_size[0]   
        #   2. agent_y / self.map_size[1]   
        #   3. sin(agent_theta)             
        #   4. cos(agent_theta)             
        #   5. has_package                  
        #   6. packages_left / self.nr_packages 
        #   7. depot_x / self.map_size[0]                               
        #   8. depot_y / self.map_size[1]
        #   9. delivery_x / self.map_size[0]
        #   10. delivery_y / self.map_size[1]
        #   11. NUM_REGIONS lidar distances 

        low = np.concatenate((
            np.array([0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0]),
            np.zeros(NUM_REGIONS)
        )).astype(np.float32)
        high = np.ones(10+NUM_REGIONS, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 7. RNG
        self.rng = np.random.default_rng(seed)

        # 8. flags
        self.hit_wall:          bool | None = None
        self.bumped_obstacle:   bool | None = None
        self.picked_up:         bool | None = None
        self.delivered:         bool | None = None

        # 9. Render mode
        self.render_mode = render_mode
        self.fig = None
        self.ax = None

    def _load_map(self, map_config: dict):
        """Check whether the provided map_config is a valid map."""
        
        # check whether indeed a dict
        if not isinstance(map_config,dict):
            raise ValueError("map_config must be a dictionary")
        
        # check whether it contains the required keys
        for key in ["size","depot","delivery","obstacles"]:
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

        # Validate & store "delivery" (must be 2 floats/ints)
        delivery = map_config.get("delivery", None)
        # sample random goal if no delivery goal specified
        if delivery is None:
            self._sample_goal()
        else:
            if (not isinstance(delivery, (tuple, list)) or len(delivery) != 2):
                raise ValueError("map_config['delivery'] must be a 2-tuple (x, y)")
            self.delivery_goal_x, self.delivery_goal_y = delivery

        # validate & store "obstacles" (list of 4-tuples)
        obstacles = map_config["obstacles"]
        if (not isinstance(obstacles, (list, tuple)) or
            any((not isinstance(o, (tuple, list)) or len(o) != 4) for o in obstacles)):
            raise ValueError("map_config['obstacles'] must be a list of 4-tuples [(xmin,ymin,xmax,ymax), …]")
        self.obstacles = np.array(obstacles, dtype=np.float32)

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
        
        # reset agent to random position and orientation
        self.agent_x, self.agent_y = self._sample_free_position()
        self.agent_theta = self.rng.uniform(-np.pi,np.pi)

        # reset environmental params
        self.has_package = False
        self.packages_left = self.nr_packages
        self.steps = 0

        # initialize flags
        self.hit_wall        = False
        self.bumped_obstacle = False
        self.picked_up       = False
        self.delivered       = False

        # init raw lidar distances (0…MAX)
        self._last_lidar = self.lidar_fixed_rays()

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

        # execute action
        match action:
            case 0: self._move(MOVE_SIZE)
            case 1: self._turn(+TURN_SIZE)
            case 2: self._turn(-TURN_SIZE)
            case _:
                raise ValueError(f"Invalid action: '{action}'.")

        reward = self._reward_fn(action)
        terminated = (self.packages_left == 0)
        self.steps+=1
        truncated = (self.steps >= self.max_steps)

        # store lidar distances
        self._last_lidar = self.lidar_fixed_rays()
        
        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        "Return a (10+NUM_REGIONS)-dimensional observation vector."

        # relative vectors to static landmarks
        vec_depot = (self.depot - [self.agent_x, self.agent_y]) / self.dmax

        # relative vectors to current goal if package is held
        if self.has_package:
            vec_goal = (np.array([self.delivery_goal_x, self.delivery_goal_y]) -
                        np.array([self.agent_x, self.agent_y])) / self.dmax
        else:
            vec_goal = np.zeros(2, dtype=np.float32)

        base = np.array([
            self.agent_x / self.map_size[0],
            self.agent_y / self.map_size[1],
            np.sin(self.agent_theta),
            np.cos(self.agent_theta),
            float(self.has_package),
            self.packages_left / self.nr_packages,
            vec_depot[0], vec_depot[1],
            vec_goal[0],  vec_goal[1],
        ], dtype=np.float32)

        # get lidar distances
        lidar = self._last_lidar / MAX_LIDAR_DISTANCE
        return np.concatenate((base, lidar)).astype(np.float32)

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
        self.picked_up       = False
        self.delivered       = False

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

        # if carrying a package, try to deliver it after moving
        if self.has_package:
            if self._try_deliver():
                self.delivered = True
        else:
            # if not carrying a package, try to pick up after moving 
            if self._try_pickup():
                self.picked_up = True


    # --- Other Action Checks ---------------------------------------

    def _legal_pickup(self) -> bool:
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

    def _legal_deliver(self) -> bool:
        return (
            self.has_package 
            and self._near([self.delivery_goal_x, self.delivery_goal_y])
        )

    def _try_deliver(self) -> bool:
        """Attempt to deliver the package from the current position. 
        Returns True iff it succeeded."""

        if self._legal_deliver():
            self.has_package = False
            self.packages_left -= 1
            return True
        return False

    # --- Random Delivery Goal Sampling -----------------------------
    def _sample_goal(self):
        """Choose a customer address with at least 0.7 m clearance from depot."""
        while True:
            gx = self.rng.uniform(0, self.map_size[0])
            gy = self.rng.uniform(0, self.map_size[1])
            if np.hypot(gx - self.depot[0], gy - self.depot[1]) > 0.7:
                # check whether inside an obstacle
                if self._in_obstacle(gx, gy):
                    continue
                # set goal
                self.delivery_goal_x, self.delivery_goal_y = gx, gy
                break
    
    # --- Random Free Position Sampling -----------------------------
    def _sample_free_position(self) -> tuple[float, float]:
        """
        Returns a random (x,y) uniformly on [0,W]x[0,H] that does NOT lie in any obstacle.
        """
        while True:
            x = self.rng.uniform(0, self.W)
            y = self.rng.uniform(0, self.H)
            if not self._in_obstacle(x, y):
                return x, y

    # --- Reward Function -------------------------------------------
    def _reward_fn(self, action: int) -> float:
        # time penalty
        r = REW_STEP

        # if agent moved
        if action == 0:
            # collision & boundary penalties
            if self.bumped_obstacle:           # set in _post_motion()
                r += REW_OBSTACLE
            if self.hit_wall:                  # set in _post_motion()
                r += REW_WALL
            if self.picked_up:                # set in _post_motion()
                r += REW_PICKUP
            if self.delivered:                # set in _post_motion()
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

    def ray_segment_intersection(self, x0, y0, dx, dy, ax, ay, bx, by):
        """
        Compute intersection of the ray (x0,y0) + t*(dx,dy), t>=0 with segment [A(ax,ay) -> B(bx,by)]
        by solving the intersection equation (x0,y0) + t*(dx,dy) = (ax,ay) + u*(bx-ax, by-ay)
        Returns t (distance along ray) if intersects within segment and t>=0, else None.
        """

        # parameterize the segment AB
        ex, ey = bx - ax, by - ay
        denom = dx*ey - dy*ex
        # return None if segment and ray are parallel or collinear
        if abs(denom) < 1e-8:
            return None
        # solve using Cramer's rule
        rx, ry = ax - x0, ay - y0
        t = (rx*ey - ry*ex) / denom
        u = (rx*dy - ry*dx) / denom
        # only accept if intersection lies in front of ray and within the segment
        if t >= 0 and 0 <= u <= 1:
            return t
        return None

    def lidar_fixed_rays(self):
        """
        Cast NUM_REGIONS rays at angles (agent_theta + k*45°), and find first obstacle or
        wall hit within MAX_LIDAR_DISTANCE if any.
        Returns: distances[k] for k=0..NUM_REGIONS-1
        """
        W, H = self.map_size
        # compile list of all segments: walls + obstacle edges
        segments = [
            (0, 0, W, 0), (W, 0, W, H), (W, H, 0, H), (0, H, 0, 0)
        ]
        # obstacle edges
        for xmin, ymin, xmax, ymax in self.obstacles:
            corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            for i in range(4):
                ax, ay = corners[i]
                bx, by = corners[(i + 1) % 4]
                segments.append((ax, ay, bx, by))

        # compute distances for each ray and select the closest intersection
        distances = np.full(NUM_REGIONS, MAX_LIDAR_DISTANCE, dtype=np.float32)
        for k in range(NUM_REGIONS):
            theta = self.agent_theta + k * REGION_FOV
            dx, dy = np.cos(theta), np.sin(theta)
            best = MAX_LIDAR_DISTANCE
            for ax, ay, bx, by in segments:
                t = self.ray_segment_intersection(
                    self.agent_x, self.agent_y, dx, dy, ax, ay, bx, by
                )
                if t is not None and t < best:
                    best = t
            distances[k] = best
        return distances

    # def lidar_wedges(self):
    #     W, H = self.map_size
    #     all_boxes = self.map_config["obstacles"] + [
    #         (0, 0, W, 0), (W, 0, W, H), (W, H, 0, H), (0, H, 0, 0)
    #     ]
    #     distances = np.full(NUM_REGIONS, MAX_LIDAR_DISTANCE, dtype=float)

    #     for xmin, ymin, xmax, ymax in all_boxes:
    #         px = np.clip(self.agent_x, xmin, xmax)
    #         py = np.clip(self.agent_y, ymin, ymax)
    #         if xmin < self.agent_x < xmax and ymin < self.agent_y < ymax:
    #             continue
    #         dx, dy = px - self.agent_x, py - self.agent_y
    #         dist = np.hypot(dx, dy)
    #         if dist == 0 or dist > MAX_LIDAR_DISTANCE:
    #             continue
    #         phi = (np.arctan2(dy, dx) % (2 * np.pi))
    #         k = int(phi // REGION_FOV)
    #         distances[k] = min(distances[k], dist)

    #     return distances

    # --- render & close --------------------------------------
    
    def render(self):
        """
        If render_mode == 'human': show a Matplotlib window that updates in real time.
        If render_mode == 'rgb_array': return a HxWx3 uint8 array of the current frame.
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
            Rectangle((self.depot[0]-0.2, self.depot[1]-0.2), 0.4*SCALE, 0.4*SCALE, color="green")
        )

        # 5. If carrying, draw current goal as a red “X”
        if self.has_package:
            gx, gy = self.delivery_goal_x, self.delivery_goal_y
            offset = 0.2 * SCALE
            self.ax.plot([gx-offset, gx+offset], [gy-offset, gy+offset], color="red", linewidth=2 * SCALE)
            self.ax.plot([gx-offset, gx+offset], [gy+offset, gy-offset], color="red", linewidth=2 * SCALE)

        # 6. Draw robot as a blue circle + arrow for orientation
        robot_circle = Circle((self.agent_x, self.agent_y), 0.3, color="blue", alpha=0.8)
        self.ax.add_patch(robot_circle)
        dx = 0.5 * np.cos(self.agent_theta)
        dy = 0.5 * np.sin(self.agent_theta)
        self.ax.arrow(self.agent_x, self.agent_y, dx, dy,
                    head_width=0.15, head_length=0.15, fc="blue", ec="blue")

        # 7. Draw package count in top-left
        pkg_text = "Carrying" if self.has_package else "Empty"
        self.ax.text(0.02 * self.W, 0.94 * self.H,
                    f"{pkg_text}, Left: {self.packages_left}", color="black",
                    fontsize=8, verticalalignment="top")

        # 8. Draw lidar rays as lines
        for k, d in enumerate(self._last_lidar):
            angle = self.agent_theta + k * REGION_FOV
            x2 = self.agent_x + d * np.cos(angle)
            y2 = self.agent_y + d * np.sin(angle)
            norm = d / MAX_LIDAR_DISTANCE
            r = 1.0 - norm
            g = norm
            self.ax.plot([self.agent_x, x2],
                        [self.agent_y, y2],
                        color=(r, g, 0),
                        linewidth=1)

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
