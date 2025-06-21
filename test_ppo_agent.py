import torch
import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from ppo_network import NetworkFactory
from environments.medium_delivery_env import MediumDeliveryEnv
from ppo_agent import PPOConfig  # Import PPOConfig to allowlist it

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
    obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)

    done = False
    while not done:
        logits = actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()

        next_obs_np, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32)).to(device)

    # Retrieve video path
    video_path = None
    if hasattr(env, "video_recorder") and env.video_recorder is not None:
        base_path = env.video_recorder.base_path
        video_path = Path(f"{base_path}.mp4")
        if not video_path.exists():
            print(f"Expected video at {video_path} but not found")
            video_path = None

    env.close()
    return video_path

def main():
    model_path = "final_model.pt"
    video_dir = "test_videos"
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MediumDeliveryEnv(render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    actor = load_model(model_path, state_size, action_size, device)
    test_model(actor, env, device, video_dir)

if __name__ == "__main__":
    main()