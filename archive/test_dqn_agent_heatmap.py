import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dqn_agent import DQNConfig, DQNAgent
from environments.medium_delivery_env import MediumDeliveryEnv
from maps import MAIL_DELIVERY_MAPS

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

def create_environment(config: DQNConfig):
    env = MediumDeliveryEnv(map_config=MAIL_DELIVERY_MAPS[config.map_name], render_mode=None, seed=42)
    return env

def load_model(model_path: str, config: DQNConfig, device: torch.device):
    env = create_environment(config)
    agent = DQNAgent(env.observation_space, env.action_space, config)
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()
    return agent

def evaluate_and_heatmap(model_path, map_name="default", trials=20, save_path="heatmap_dqn.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DQNConfig(map_name=map_name)
    env = create_environment(config)
    agent = load_model(model_path, config, device)
    agent.epsilon = 0.0  # Ensure greedy policy for evaluation

    grid_size = 100
    map_w, map_h = env.map_size
    heatmap = np.zeros((grid_size, grid_size), dtype=np.int32)

    for episode in range(trials):
        obs_np, _ = env.reset()
        done = False
        step_count = 0
        total_score = 0
        while not done:
            x = getattr(env, "agent_x", None)
            y = getattr(env, "agent_y", None)
            if step_count == 0:
                print(f"Initial position: x={x:.4f}, y={y:.4f}")
            grid_x = int(np.clip(x / map_w * (grid_size - 1), 0, grid_size - 1))
            grid_y = int(np.clip(y / map_h * (grid_size - 1), 0, grid_size - 1))
            heatmap[grid_y, grid_x] += 1
            state = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)
            if hasattr(agent, 'get_action'):
                action = agent.get_action(state.cpu().numpy())
            else:
                q_values = agent.model(state.unsqueeze(0))
                action = torch.argmax(q_values, dim=1).item()
            next_obs_np, score, terminated, truncated, _ = env.step(action)
            total_score += score
            done = terminated or truncated
            obs_np = next_obs_np
            step_count += 1
        print(f"Episode {episode + 1}/{trials} - Steps: {step_count}, Score: {total_score}")

    env.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="hot", interpolation="nearest", origin="lower", extent=[0, map_w, 0, map_h])
    plt.title(f"DQN Agent Heatmap of Visited Spaces ({trials} episodes)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Visit Count")
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    evaluate_and_heatmap(
        model_path="checkpoints_dqn/model_final_model_hq6fgyw7.pth",
        map_name="default",
        trials=5,
        save_path="dqn_agent_heatmap.png"
    )
