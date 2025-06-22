import torch
import numpy as np
from pathlib import Path
from gymnasium.wrappers import RecordVideo
from dqn_agent import DQNConfig, DQNAgent
from environments.medium_delivery_env import MediumDeliveryEnv
from maps import MAIL_DELIVERY_MAPS

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

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

def test_model(agent, env, device: torch.device, video_dir: str = None, video_name: str = None):
    if video_dir:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True, 
                          name_prefix=video_name if video_name else "dqn_test")
    obs_np, _ = env.reset()
    print(f"Observation space: x-{obs_np[0]:.4f}, y-{obs_np[1]:.4f}")
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
        video_path = Path(env.video_recorder.path)
        if not video_path.exists():
            print(f"Expected video at {video_path} but not found")
            video_path = None
    env.close()
    return total_reward, video_path

def main():
    map = "empty"
    trials = 50
    model_path = "checkpoints_dqn/model_final_model.pth"
    video_dir = f"videos/test_videos_dqn_{map}"

    Path(video_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DQNConfig(map_name="empty")
    env = create_environment(config)
    agent = load_model(model_path, config, device)
    for i in range(1, trials + 1):
        total_reward, video_path = test_model(agent, env, device, video_dir, video_name=f"{map}_{i}")
        print(f"Test episode {i} reward: {total_reward}")
        if video_path:
            print(f"Video saved at: {video_path}")

if __name__ == "__main__":
    main()
