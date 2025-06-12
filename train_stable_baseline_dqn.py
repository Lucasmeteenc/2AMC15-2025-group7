import os

# Import necessary libraries
import torch
import wandb
from wandb.integration.sb3 import WandbCallback # Import W&B callback for SB3

# stable_baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback



# --- define env ----------------------------------------------------
# Possible values: "CustomEnv" (from environment.py), "SimpleDeliveryEnv" (from environment2.py)
ENV_NAME = "SimpleDeliveryEnv"

# from environment2 import SimpleDeliveryEnv
from environments.simple_delivery_env import SimpleDeliveryEnv

def train_dqn(total_timesteps=pow(10,6), project_name="custom_robot_rl", run_name=None):
    """
    Train a DQN agent on the given environment.

    :param env: The environment to train on.
    :param total_timesteps: Total number of timesteps for training.
    :return: The trained DQN model.
    """

    log_dir = 'results/logs/'
    model_save_dir = os.path.join(log_dir, 'dqn_models_local')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # --- try to run on GPU if possible ---------------------------------
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- W&B Configuration ---
    config = {
        "env_name": ENV_NAME,
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "learning_rate": 1e-4, # default DQN lr in SB3 is 1e-4 (see Nature DQN paper) 
        "buffer_size": 100000,
        "learning_starts": 10000,
        "batch_size": 64,
        "tau": 0.005, # 1.0: any update to target network fully overwrites the online network
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": -1, # # -1: do as many gradient steps as train_freq
        "exploration_fraction": 0.3, # fraction of training period over which the exploration rate is reduced
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy_kwargs": dict(net_arch=[128, 128]),
        "device": device,
    }

    # --- Initialize W&B Run ---
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        sync_tensorboard=True,  # Automatically sync SB3's TensorBoard logs
        monitor_gym=False,       #! Automatically log videos of the environment -> set to False bc VecVideoRecorder also records videos
        save_code=True,         # Save the main script to W&B
    )

    # This is especially useful if sync_tensorboard=True
    tensorboard_log_path = os.path.join(log_dir, f"tensorboard_logs_{run.id}")
    os.makedirs(tensorboard_log_path, exist_ok=True)

    def make_env():
        # setup env
        env = SimpleDeliveryEnv(render_mode='rgb_array')

        # setup csv for monitoring
        os.makedirs(log_dir, exist_ok=True)
        monitor_file = os.path.join(log_dir, f"monitor_{run.id}.csv")
        env = Monitor(env, filename=monitor_file) #! change: Monitor expects a file name, not a directory
        return env
    
    vec_env = DummyVecEnv([make_env])

    # VERY SLOW RECORDING. reduced to 150 frames.
    vec_env = VecVideoRecorder(
        vec_env,
        f"results/videos/{run.id}",
        record_video_trigger=lambda x: x % 20000 == 0, # Record every 20000 steps --> this does mean that the video at the end of training will only be one frame
        video_length=500 # Max length of recorded video
    )

    model = DQN(
        config['policy_type'],
        vec_env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        exploration_fraction=config['exploration_fraction'],
        exploration_initial_eps=config['exploration_initial_eps'],
        exploration_final_eps=config['exploration_final_eps'],
        policy_kwargs=config['policy_kwargs'],
        verbose=0,
        tensorboard_log=tensorboard_log_path,
        device=config['device'],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000 // vec_env.num_envs, 1),
        save_path=model_save_dir,
        name_prefix=f'dqn_model_local_{run.id}',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=10000,
        model_save_path=f"models/{run.id}",
        model_save_freq=max(20000 // vec_env.num_envs, 1),
        log="all",
        verbose=2,
    )

    callbacks = [checkpoint_callback, wandb_callback]

    print("Starting training...")
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            log_interval=10,
        )
    finally:
        run.finish()

    print("Training complete. Final model saving locally.")
    final_model_path = os.path.join(log_dir, f"dqn_model_final_{run.id}")
    model.save(final_model_path)

    vec_env.close()

    return model

def test_dqn(num_episodes=10, model_path=None):
    """
    Test the trained DQN model.
    :param model: The trained DQN model.
    :param num_episodes: Number of episodes to test.
    """

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    model = DQN.load(model_path)
    print(f"Loaded model from {model_path}")

    test_env = SimpleDeliveryEnv(render_mode='human')
    
    observation, _ = test_env.reset()

    print(f"\n--- Testing for {num_episodes} episodes ---")
    for episode in range(num_episodes):
        terminated = False
        truncated = False

        total_reward = 0
        step = 0

        observation, _ = test_env.reset()

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = test_env.step(action)

            total_reward += reward
            step += 1

            test_env.render()

            if terminated or truncated:
                print(f"Episode {episode + 1} finished after {step} steps with total reward: {total_reward}")
                if info.get("is_success"):
                    print("Agent successfully delivered the package!")
                break

    test_env.close()


if __name__ == "__main__":
    trained_model = train_dqn(total_timesteps=pow(10,6))
    # test_dqn(num_episodes=10, model_path='models/8vu65zod/model.zip')