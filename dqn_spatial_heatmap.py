import argparse
import logging
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
import torch

from dqn_agent import DQNAgent, DQNConfig
from environments.medium_delivery_env import MediumDeliveryEnv
from maps import MAIL_DELIVERY_MAPS

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
FIXED_START_POSITION = (8.0, 4.0)   # depot coordinates
FIXED_START_THETA = 0
SCALE                 = 1.0

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def render_map(ax, map_config):
    """Draw the map background, obstacles, depot, and delivery point."""
    width, height = map_config["size"]
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])

    # Obstacles
    for x0, y0, x1, y1 in map_config["obstacles"]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               color="gray", alpha=0.5))

    # Depot (green square)
    depot_x, depot_y = map_config["depot"]
    ax.add_patch(Rectangle((depot_x - 0.2 * SCALE,
                            depot_y - 0.2 * SCALE),
                           0.4 * SCALE, 0.4 * SCALE,
                           color="green"))
    # Delivery point (red X)
    deliv_x, deliv_y = map_config["delivery"]
    off = 0.2 * SCALE
    ax.plot([deliv_x - off, deliv_x + off],
            [deliv_y - off, deliv_y + off],
            color="red", linewidth=2 * SCALE)
    ax.plot([deliv_x - off, deliv_x + off],
            [deliv_y + off, deliv_y - off],
            color="red", linewidth=2 * SCALE)


def create_environment(cfg: DQNConfig):
    """Instantiate the MediumDelivery environment with a fixed seed."""
    return MediumDeliveryEnv(map_config=MAIL_DELIVERY_MAPS[cfg.map_name],
                             render_mode=None, seed=42)


def load_model(path: str, cfg: DQNConfig, device: torch.device):
    """Load a pre-trained model (if present) into a freshly-built agent."""
    env   = create_environment(cfg)
    agent = DQNAgent(env.observation_space, env.action_space, cfg)

    # if Path(path).is_file():
    #     agent.model.load_state_dict(torch.load(path, map_location=device))
    # else:
    #     logging.warning("Model file not found at %s – using untrained network",
    #                     path)
    if not Path(path).is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    agent.model.load_state_dict(torch.load(path, map_location=device))

    agent.model.to(device).eval()
    return agent


def evaluate_and_spatial_heatmap(model_path: str,
                                 map_name: str,
                                 trials: int,
                                 bandwidth: float,
                                 cmap_name: str,
                                 save_path: str):
    """Run multiple evaluation episodes and plot a KDE heat-map."""
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg     = DQNConfig(map_name=map_name)
    env     = create_environment(cfg)
    agent   = load_model(model_path, cfg, device)
    agent.epsilon = 0.0              # greedy policy

    positions = []

    for ep in range(trials):
        obs_np, _      = env.reset()
        env.agent_x, env.agent_y = FIXED_START_POSITION   # teleport to depot
        env.agent_theta = FIXED_START_THETA
        obs_np = env._get_obs()

        done, step_cnt, score_sum = False, 0, 0.0
        while not done:
            # -- record position BEFORE we move (optionally capture after move)
            positions.append((env.agent_x, env.agent_y))

            state = torch.from_numpy(np.asarray(obs_np, np.float32)).to(device)

            # -- choose an action
            if hasattr(agent, "get_action"):
                action = agent.get_action(state.cpu().numpy())
            else:
                with torch.no_grad():
                    q_vals = agent.model(state.unsqueeze(0))
                    action = torch.argmax(q_vals, dim=1).item()

            # -- environment step
            next_obs_np, reward, terminated, truncated, _ = env.step(action)
            score_sum += reward
            done       = terminated or truncated
            obs_np     = next_obs_np
            step_cnt  += 1

        logging.info("Episode %d/%d - steps: %d, score: %.2f",
                     ep + 1, trials, step_cnt, score_sum)

    env.close()

    # --------------------------------------------------------------------- #
    # Plot heat-map
    # --------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 8))
    render_map(ax, MAIL_DELIVERY_MAPS[map_name])

    if positions:
        xs, ys   = zip(*positions)
        cmap     = plt.get_cmap(cmap_name)

        sns.kdeplot(x=xs, y=ys,
                    cmap=cmap,
                    fill=True,
                    alpha=0.6,
                    n_levels=100,
                    bw_adjust=bandwidth,
                    ax=ax)

    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(f"DQN Agent Continuous Spatial Heat-map ({trials} episodes)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info("Spatial heat-map saved to %s", save_path)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Visualise DQN spatial heat-map")
    parser.add_argument("--model",  default="best_models/default/dqn.pth",
                        help="Path to the .pth model file")
    parser.add_argument("--map",    default="default",
                        choices=list(MAIL_DELIVERY_MAPS.keys()),
                        help="Map name")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--bw",     type=float, default=0.5,
                        help="Bandwidth scaling for KDE (lower=hug path)")
    parser.add_argument("--cmap",   default="viridis",
                        help="Matplotlib colour-map name (e.g. jet,viridis, magma)")
    parser.add_argument("--out",    default="dqn_spatial_heatmap.png",
                        help="Output PNG path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase logging verbosity")
    args = parser.parse_args()

    # Configure warnings and logging
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    evaluate_and_spatial_heatmap(model_path=args.model,
                                 map_name=args.map,
                                 trials=args.trials,
                                 bandwidth=args.bw,
                                 cmap_name=args.cmap,
                                 save_path=args.out)


if __name__ == "__main__":
    main()


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, Circle
# import seaborn as sns
# from pathlib import Path
# from dqn_agent import DQNConfig, DQNAgent
# from environments.medium_delivery_env import MediumDeliveryEnv

# import argparse

# from maps import MAIL_DELIVERY_MAPS

# # Ignore all warnings
# import warnings
# warnings.filterwarnings("ignore")

# # Fixed starting position (hardcoded)
# FIXED_START_POSITION = (5.0, 1.5)  # Depot coordinates

# SCALE = 1.0
# # MAX_LIDAR_DISTANCE = 10.0

# def render_map(ax, map_config):
#     """Renders the map background on a given matplotlib axis."""
#     W, H = map_config["size"]
#     ax.set_xlim(0, W)
#     ax.set_ylim(0, H)
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_title("DQN Agent Spatial Heatmap")
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # Draw obstacles
#     for obs in map_config["obstacles"]:
#         rect = Rectangle((obs[0], obs[1]), obs[2] - obs[0], obs[3] - obs[1],
#                          color="gray", alpha=0.5)
#         ax.add_patch(rect)

#     # Draw depot
#     depot = map_config["depot"]
#     ax.add_patch(
#         Rectangle((depot[0] - 0.2 * SCALE, depot[1] - 0.2 * SCALE), 0.4 * SCALE, 0.4 * SCALE, color="green")
#     )

#     # Draw delivery
#     delivery = map_config["delivery"]
#     offset = 0.2 * SCALE
#     ax.plot([delivery[0] - offset, delivery[0] + offset], [delivery[1] - offset, delivery[1] + offset], color="red", linewidth=2 * SCALE)
#     ax.plot([delivery[0] - offset, delivery[0] + offset], [delivery[1] + offset, delivery[1] - offset], color="red", linewidth=2 * SCALE)


# def create_environment(config: DQNConfig):
#     """Creates the medium delivery environment."""
#     env = MediumDeliveryEnv(map_config=MAIL_DELIVERY_MAPS[config.map_name], render_mode=None, seed=42)
#     return env

# def load_model(model_path: str, config: DQNConfig, device: torch.device):
#     """Loads a pre-trained DQN agent model."""
#     env = create_environment(config)
#     agent = DQNAgent(env.observation_space, env.action_space, config)
#     # Ensure the model exists before trying to load it
#     if Path(model_path).exists():
#         agent.model.load_state_dict(torch.load(model_path, map_location=device))
#     else:
#         print(f"Warning: Model file not found at {model_path}. A new untrained model will be used.")
#     agent.model.eval()
#     return agent

# def evaluate_and_spatial_heatmap(model_path, map_name="default", trials=20, save_path="dqn_spatial_heatmap.png"):
#     """Generates a continuous spatial heatmap of the agent's movement."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config = DQNConfig(map_name=map_name)
#     env = create_environment(config)
#     agent = load_model(model_path, config, device)
#     agent.epsilon = 0.0  # Greedy policy for evaluation

#     positions = []

#     for episode in range(trials):
#         obs_np, _ = env.reset()
#         env.agent_x, env.agent_y = FIXED_START_POSITION
        
#         done = False
#         step_count = 0
#         total_score = 0
#         while not done:
#             x = getattr(env, "agent_x", None)
#             y = getattr(env, "agent_y", None)
#             if x is not None and y is not None:
#                 positions.append((x, y))

#             state = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)
#             if hasattr(agent, 'get_action'):
#                 action = agent.get_action(state.cpu().numpy())
#             else:
#                 q_values = agent.model(state.unsqueeze(0))
#                 action = torch.argmax(q_values, dim=1).item()

#             next_obs_np, score, terminated, truncated, _ = env.step(action)
#             total_score += score
#             done = terminated or truncated
#             obs_np = next_obs_np
#             step_count += 1
#         print(f"Episode {episode + 1}/{trials} - Steps: {step_count}, Score: {total_score}")

#     env.close()

#     # Create the plot
#     fig, ax = plt.subplots(figsize=(8, 8))
    
#     # Render the map background
#     render_map(ax, MAIL_DELIVERY_MAPS[map_name])

#     # Overlay the heatmap using Kernel Density Estimation
#     if positions:
#         pos_x, pos_y = zip(*positions)
#         # Explicitly get the colormap from matplotlib to avoid the ValueError
#         jet_cmap = plt.get_cmap("jet")

#         # KDE plot with reduced smoothing bandwidth (to make it hug the path)
#         sns.kdeplot(
#             x=pos_x, y=pos_y,
#             cmap=jet_cmap,
#             fill=True,
#             ax=ax,
#             alpha=0.6,
#             n_levels=100,
#             bw_adjust=0.5,   # Adjust the bandwidth for less smoothing
#             shade=True       # Enable shading to create a soft gradient without contours
#         )

#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title(f"DQN Agent Continuous Spatial Heatmap ({trials} episodes)")
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Spatial heatmap saved to {save_path}")

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(
#         description="DQN spatial heatmap"
#     )
#     parser.add_argument("--model",  type=str,
#                         default="best_models/dqn.pth")
#     parser.add_argument("--map",    type=str,
#                         default="default",
#                         choices=list(MAIL_DELIVERY_MAPS.keys()))
#     parser.add_argument("--trials", type=int,   default=100,
#                         help="Number of episodes")
#     # parser.add_argument("--grid",   type=int,   default=300,
#     #                     help="Histogram resolution")
#     # parser.add_argument("--sigma",  type=float, default=1.2,
#     #                     help="Gaussian blur sigma (smaller=tighter)")
#     # parser.add_argument("--threshold", type=float, default=0.005,
#     #                     help="Mask out values < threshold_frac*max")
#     parser.add_argument("--out",    type=str,   default="dqn_spatial_heatmap.png")
#     args = parser.parse_args()


#     # In a real scenario, this would be a pre-trained model file.
#     # model_path = "checkpoints_dqn/model_final_model_15rngm8u.pth"
#     # if not Path(model_path).exists():
#     #     Path(model_path).parent.mkdir(parents=True, exist_ok=True)
#     #     # This part assumes DQNConfig and DQNAgent are defined elsewhere
#     #     try:
#     #         config = DQNConfig(map_name="default")
#     #         env = create_environment(config)
#     #         agent = DQNAgent(env.observation_space, env.action_space, config)
#     #         torch.save(agent.model.state_dict(), model_path)
#     #         print(f"Created a dummy model at {model_path} for demonstration purposes.")
#     #     except NameError as e:
#     #         print(f"Could not create a dummy model due to undefined components: {e}")
#     #         pass
            
#     evaluate_and_spatial_heatmap(
#         model_path=args.model,
#         map_name=args.map,
#         trials=args.trials,
#         save_path=args.out
#     )









    
# import argparse
# from pathlib import Path

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter
# from matplotlib.patches import Rectangle

# from dqn_agent import DQNConfig, DQNAgent
# from environments.medium_delivery_env import MediumDeliveryEnv
# from maps import MAIL_DELIVERY_MAPS

# # ------------------------------------------------------------------------------
# # 1. Environment & Model Utilities
# # ------------------------------------------------------------------------------

# def make_env(map_name: str, seed: int = 42) -> MediumDeliveryEnv:
#     cfg = DQNConfig(map_name=map_name)
#     return MediumDeliveryEnv(
#         map_config=MAIL_DELIVERY_MAPS[cfg.map_name],
#         render_mode=None,
#         seed=seed
#     )

# def load_agent(model_path: str,
#                config: DQNConfig,
#                env: MediumDeliveryEnv,
#                device: torch.device) -> DQNAgent:
#     agent = DQNAgent(env.observation_space, env.action_space, config)
#     agent.model.to(device)
#     ckpt = torch.load(model_path, map_location=device)
#     agent.model.load_state_dict(ckpt)
#     agent.model.eval()
#     agent.epsilon = 0.0
#     agent.device = device
#     return agent

# # ------------------------------------------------------------------------------
# # 2. Rollout and Accumulate Visits (Histogram)
# # ------------------------------------------------------------------------------

# def collect_visit_counts(agent: DQNAgent,
#                          env: MediumDeliveryEnv,
#                          trials: int = 50,
#                          grid_size: int = 300) -> np.ndarray:
#     W, H = env.map_size
#     counts = np.zeros((grid_size, grid_size), dtype=np.int32)

#     for ep in range(trials):
#         obs, _ = env.reset()
#         done = False
#         while not done:
#             x, y = env.agent_x, env.agent_y
#             i = int(np.clip(x / W * (grid_size-1), 0, grid_size-1))
#             j = int(np.clip(y / H * (grid_size-1), 0, grid_size-1))
#             counts[j, i] += 1

#             state = torch.from_numpy(obs.astype(np.float32)).to(agent.device)
#             q = agent.model(state.unsqueeze(0))
#             action = int(q.argmax(dim=1).item())

#             obs, _, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#         print(f"› Episode {ep+1}/{trials} done.")
#     env.close()
#     return counts

# # ------------------------------------------------------------------------------
# # 3. Draw Map + Overlay Histogram Heatmap with Threshold Masking
# # ------------------------------------------------------------------------------

# def plot_histogram_heatmap(counts: np.ndarray,
#                            map_config: dict,
#                            smooth_sigma: float = 1.2,
#                            threshold_frac: float = 0.005,
#                            cmap: str = "jet",
#                            alpha: float = 0.8,
#                            save_path: str = "dqn_spatial_heatmap.png"):
#     """
#     - Smooth the visit-count histogram with gaussian_filter
#     - Mask out any cell below threshold_frac * max(smooth)
#     - Overlay on the map, clamping colormap to [threshold, max]
#     """
#     # 1) apply Gaussian blur
#     smooth = gaussian_filter(counts.astype(np.float32), sigma=smooth_sigma)

#     # 2) threshold mask
#     vmin = smooth.max() * threshold_frac
#     smooth_masked = np.ma.masked_where(smooth < vmin, smooth)

#     # 3) draw base map
#     W, H = map_config["size"]
#     fig, ax = plt.subplots(figsize=(6,6))
#     ax.set_xlim(0, W); ax.set_ylim(0, H)
#     ax.set_aspect("equal", "box")
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.set_title("DQN Visit Density (hist+blur, jet)")

#     # obstacles
#     for xmin, ymin, xmax, ymax in map_config["obstacles"]:
#         ax.add_patch(Rectangle((xmin, ymin),
#                                xmax-xmin, ymax-ymin,
#                                color="gray", alpha=0.5))

#     # depot
#     dx, dy = map_config["depot"]
#     s = 0.3
#     ax.add_patch(Rectangle((dx-s, dy-s), 2*s, 2*s, color="green"))

#     # delivery
#     gx, gy = map_config["delivery"]
#     off = 0.3
#     ax.plot([gx-off, gx+off], [gy-off, gy+off], color="red", lw=2)
#     ax.plot([gx-off, gx+off], [gy+off, gy-off], color="red", lw=2)

#     # 4) overlay heatmap, clamp to [vmin, vmax]
#     im = ax.imshow(
#         smooth_masked,
#         origin="lower",
#         extent=[0, W, 0, H],
#         interpolation="bilinear",
#         cmap=cmap,
#         alpha=alpha,
#         vmin=vmin,
#         vmax=smooth.max()
#     )

#     # 5) colorbar
#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label("Smoothed Visit Count")

#     # 6) save
#     out = Path(save_path)
#     out.parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(out, dpi=300)
#     plt.close(fig)
#     print(f"Saved histogram-based heatmap to {out.resolve()}")

# # ------------------------------------------------------------------------------
# # 4. Main
# # ------------------------------------------------------------------------------

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="DQN spatial heatmap (histogram + blur + threshold)"
#     )
#     parser.add_argument("--model",  type=str,
#                         default="checkpoints_dqn/model_final_model_d9tr8v60.pth")
#     parser.add_argument("--map",    type=str,
#                         default="default",
#                         choices=list(MAIL_DELIVERY_MAPS.keys()))
#     parser.add_argument("--trials", type=int,   default=50,
#                         help="Number of episodes")
#     parser.add_argument("--grid",   type=int,   default=300,
#                         help="Histogram resolution")
#     parser.add_argument("--sigma",  type=float, default=1.2,
#                         help="Gaussian blur sigma (smaller=tighter)")
#     parser.add_argument("--threshold", type=float, default=0.005,
#                         help="Mask out values < threshold_frac*max")
#     parser.add_argument("--out",    type=str,   default="dqn_spatial_heatmap.png")
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cfg    = DQNConfig(map_name=args.map)
#     env    = make_env(args.map)
#     agent  = load_agent(args.model, cfg, env, device)

#     counts = collect_visit_counts(agent, env,
#                                   trials=args.trials,
#                                   grid_size=args.grid)

#     plot_histogram_heatmap(counts,
#                            MAIL_DELIVERY_MAPS[args.map],
#                            smooth_sigma=args.sigma,
#                            threshold_frac=args.threshold,
#                            cmap="jet",
#                            alpha=0.8,
#                            save_path=args.out)