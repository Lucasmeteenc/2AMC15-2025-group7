import torch
import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from dqn_agent import DQNConfig, DQNAgent
from environments.medium_delivery_env import MediumDeliveryEnv
from maps import MAIL_DELIVERY_MAPS

def create_environment(config: DQNConfig):
    env = MediumDeliveryEnv(map_config=MAIL_DELIVERY_MAPS[config.map_name], render_mode="rgb_array")
    return env

# Helper to load a trained DQN model
def load_model(model_path: str, config: DQNConfig, device: torch.device):
    env = create_environment(config)
    agent = DQNAgent(env.observation_space, env.action_space, config)
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()
    return agent

def test_model(agent, env, device: torch.device, video_dir: str = None):
    if video_dir:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True, name_prefix="dqn_test")
    obs_np, _ = env.reset()
    done = False
    total_reward = 0.0
    agent.epsilon = 0.0  # Greedy policy
    while not done:
        state = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)
        if hasattr(agent, 'get_action'):
            action = agent.get_action(state.cpu().numpy())
        else:
            q_values = agent.model(state.unsqueeze(0))
            action = torch.argmax(q_values, dim=1).item()
        next_obs_np, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        obs_np = next_obs_np
    # Retrieve video path
    video_path = None
    if hasattr(env, "video_recorder") and env.video_recorder is not None:
        base_path = env.video_recorder.base_path
        video_path = Path(f"{base_path}.mp4")
        if not video_path.exists():
            print(f"Expected video at {video_path} but not found")
            video_path = None
    env.close()
    return total_reward, video_path

def main():
    model_path = "checkpoints/model_383.0.pth"
    video_dir = "test_videos_dqn_empty"
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DQNConfig(map_name="empty")
    env = create_environment(config)
    agent = load_model(model_path, config, device)
    total_reward, video_path = test_model(agent, env, device, video_dir)
    print(f"Test episode reward: {total_reward}")
    if video_path:
        print(f"Video saved at: {video_path}")

if __name__ == "__main__":
    main()
