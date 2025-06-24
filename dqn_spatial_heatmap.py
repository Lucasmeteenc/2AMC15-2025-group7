#!/usr/bin/env python3
# dqn_spatial_heatmap.py

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle

from dqn_agent import DQNConfig, DQNAgent
from environments.medium_delivery_env import MediumDeliveryEnv
from maps import MAIL_DELIVERY_MAPS

# ------------------------------------------------------------------------------
# 1. Environment & Model Utilities
# ------------------------------------------------------------------------------

def make_env(map_name: str, seed: int = 42) -> MediumDeliveryEnv:
    cfg = DQNConfig(map_name=map_name)
    return MediumDeliveryEnv(
        map_config=MAIL_DELIVERY_MAPS[cfg.map_name],
        render_mode=None,
        seed=seed
    )

def load_agent(model_path: str,
               config: DQNConfig,
               env: MediumDeliveryEnv,
               device: torch.device) -> DQNAgent:
    agent = DQNAgent(env.observation_space, env.action_space, config)
    agent.model.to(device)
    ckpt = torch.load(model_path, map_location=device)
    agent.model.load_state_dict(ckpt)
    agent.model.eval()
    agent.epsilon = 0.0
    agent.device = device
    return agent

# ------------------------------------------------------------------------------
# 2. Collect Raw Positions
# ------------------------------------------------------------------------------

def collect_positions(agent: DQNAgent,
                      env: MediumDeliveryEnv,
                      trials: int = 50) -> np.ndarray:
    """
    Run `trials` episodes, record every continuous (x,y) of the agent.
    Returns a 2×N array of positions.
    """
    all_xy = []
    for ep in range(trials):
        obs, _ = env.reset()
        done = False
        while not done:
            x, y = env.agent_x, env.agent_y
            all_xy.append((x, y))

            state = torch.from_numpy(obs.astype(np.float32)).to(agent.device)
            q = agent.model(state.unsqueeze(0))
            action = int(q.argmax(dim=1).item())

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        print(f"› Episode {ep+1}/{trials} done.")
    env.close()
    return np.array(all_xy).T  # shape (2, N)

# ------------------------------------------------------------------------------
# 3. Plot Continuous KDE Heatmap
# ------------------------------------------------------------------------------

def plot_kde_heatmap(xy: np.ndarray,
                     map_config: dict,
                     grid_size: int = 300,
                     bw_method: float = 0.05,
                     clip_frac: float = 0.01,
                     cmap: str = "jet",
                     alpha: float = 0.6,
                     save_path: str = "dqn_spatial_heatmap.png"):
    """
    xy: 2×N array of samples
    bw_method: passed to gaussian_kde (smaller → tighter blobs)
    clip_frac: mask out densities below clip_frac * max_density
    """
    # 1) fit KDE
    kde = gaussian_kde(xy, bw_method=bw_method)

    # 2) grid to evaluate on
    W, H = map_config["size"]
    xi = np.linspace(0, W, grid_size)
    yi = np.linspace(0, H, grid_size)
    xx, yy = np.meshgrid(xi, yi)
    pts = np.vstack([xx.ravel(), yy.ravel()])

    # 3) evaluate density
    zz = kde(pts).reshape(grid_size, grid_size)

    # 4) mask out low-density fringes
    thr = zz.max() * clip_frac
    zz_masked = np.ma.masked_where(zz < thr, zz)

    # 5) draw map
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal", "box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("DQN Visit Density (KDE, jet)")

    # obstacles
    for xmin, ymin, xmax, ymax in map_config["obstacles"]:
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               color="gray", alpha=0.5))

    # depot (green square)
    dx, dy = map_config["depot"]
    s = 0.3
    ax.add_patch(Rectangle((dx-s, dy-s), 2*s, 2*s, color="green"))

    # delivery (red X)
    gx, gy = map_config["delivery"]
    off = 0.3
    ax.plot([gx-off, gx+off], [gy-off, gy+off], color="red", lw=2)
    ax.plot([gx-off, gx+off], [gy+off, gy-off], color="red", lw=2)

    # 6) overlay heatmap
    im = ax.imshow(zz_masked,
                   origin="lower",
                   extent=[0, W, 0, H],
                   interpolation="bilinear",
                   cmap=cmap,
                   alpha=alpha)

    # 7) colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Estimated Density")

    # 8) save
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved KDE heatmap to {out.resolve()}")

# ------------------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DQN spatial heatmap via KDE"
    )
    parser.add_argument("--model",  type=str,
                        default="checkpoints_dqn/model_final_model_9ora6ung.pth",
                        help="Path to your .pth model")
    parser.add_argument("--map",    type=str,
                        default="default",
                        choices=list(MAIL_DELIVERY_MAPS.keys()))
    parser.add_argument("--trials", type=int,   default=50,
                        help="Number of episodes to roll out")
    parser.add_argument("--grid",   type=int,   default=300,
                        help="Resolution of the evaluation grid")
    parser.add_argument("--bw",     type=float, default=0.05,
                        help="KDE bandwidth (smaller→tighter blobs)")
    parser.add_argument("--clip",   type=float, default=0.01,
                        help="Mask densities below clip*max_density")
    parser.add_argument("--out",    type=str,   default="dqn_spatial_heatmap.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = DQNConfig(map_name=args.map)
    env    = make_env(args.map)
    agent  = load_agent(args.model, cfg, env, device)

    xy = collect_positions(agent, env, trials=args.trials)

    plot_kde_heatmap(xy,
                     MAIL_DELIVERY_MAPS[args.map],
                     grid_size=args.grid,
                     bw_method=args.bw,
                     clip_frac=args.clip,
                     cmap="jet",
                     alpha=0.6,
                     save_path=args.out)
