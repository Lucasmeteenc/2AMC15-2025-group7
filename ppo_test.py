import torch
import numpy as np
from pathlib import Path
from gymnasium.wrappers import RecordVideo
from ppo_network import NetworkFactory
from environments.medium_delivery_env import MediumDeliveryEnv
from ppo_agent import PPOConfig  # Import PPOConfig to allowlist it
from maps import MAIL_DELIVERY_MAPS

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# Allowlist PPOConfig for safe deserialization
torch.serialization.add_safe_globals([PPOConfig])

def load_model(model_path: str, state_size: int, action_size: int, device: torch.device):
    """Load the trained PPO model."""
    actor = NetworkFactory.create_actor(state_size, action_size).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor

def test_model(actor, env, device: torch.device, video_dir: str):
    """Test the PPO model and save the video."""
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True, name_prefix="ppo_test")
    obs_np, _ = env.reset()
    video_name = env._video_name
    obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)

    total_reward = 0.0
    done = False
    while not done:
        logits = actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()

        next_obs_np, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        obs = torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32)).to(device)

    # Retrieve video path
    video_path = Path(video_dir) / f"{video_name}.mp4"
    env.close()
    return total_reward, video_path

def main():
    map = "default"
    trials = 10
    model_path = "checkpoints_ppo/final_model_default.pth"
    video_dir = f"videos/test_videos_ppo_{map}"
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MediumDeliveryEnv(map_config=MAIL_DELIVERY_MAPS["default"], render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    actor = load_model(model_path, state_size, action_size, device)
    
    for i in range(1, trials + 1):
        total_reward, video_path = test_model(actor, env, device, video_dir)
        print(f"Test episode {i} reward: {total_reward}")
        if video_path:
            print(f"Video saved at: {video_path}")

if __name__ == "__main__":
    main()