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
FIXED_START_POSITION = (9.0, 4.0)  # interesting coords: (9.0,4.0), theta=0
FIXED_START_THETA = 0
SCALE = 1.0

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
    
    # mark the fixed start position with a small blue circle
    start_x, start_y = FIXED_START_POSITION
    ax.plot(start_x, start_y,
            marker='o',
            markersize=6,     # smaller than depot square
            color='lightblue',
            # zorder=10,        # ensure it sits on top
            label='Start')

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
        env.agent_x, env.agent_y = FIXED_START_POSITION
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
                        help="Bandwidth scaling for KDE (lower=hug path)") # width around agent path
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