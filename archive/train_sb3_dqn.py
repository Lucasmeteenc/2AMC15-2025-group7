import os
import random
import numpy as np
import torch
import wandb

# Stable Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback


# --- define environment --------------------------------------------
# possible values: "SimpleDeliveryEnv", "MediumDeliveryEnv", "ComplexDeliveryEnv"
ENV_NAME = "SimpleDeliveryEnv"
MAP_NAME = "default"
TOTAL_TIME_STEPS = 5 * pow(10,6)
# -------------------------------------------------------------------

# import environments and maps
from environments.simple_delivery_env  import SimpleDeliveryEnv
from environments.medium_delivery_env   import MediumDeliveryEnv
from environments.complex_delivery_env  import ComplexDeliveryEnv
from maps import MAIL_DELIVERY_MAPS, COMPLEX_DELIVERY_MAPS

ENV_REGISTRY = {
    "SimpleDeliveryEnv" : (SimpleDeliveryEnv, MAIL_DELIVERY_MAPS),
    "MediumDeliveryEnv" : (MediumDeliveryEnv, MAIL_DELIVERY_MAPS),
    "ComplexDeliveryEnv": (ComplexDeliveryEnv, COMPLEX_DELIVERY_MAPS),
}

# --- Set Seed for Reproducibility ----------------------------------
# !DONT USE THIS: IMPLEMENTED INCORRECTLY.
SEED = None#42
if SEED:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def resolve_env(env_name: str = ENV_NAME):
    """
    Return the env class and its map dictionary from ENV_REGISTRY.
    """
    try:
        return ENV_REGISTRY[env_name]
    except KeyError as e:
        raise ValueError(
            f"Unknown ENV_NAME '{env_name}'. "
            f"Valid options: {list(ENV_REGISTRY)}"
        ) from e

class EpsilonLoggerCallback(BaseCallback):
    """
    A custom callback to log the exploration rate (epsilon) to W&B.
    """
    def _on_step(self) -> bool:
        # The exploration rate is part of the model's internals
        epsilon = self.model.exploration_rate
        self.logger.record("rollout/ep_exploration_rate", epsilon)
        return True

def train_dqn(project_name="SB3-DQN", run_name=None, total_timesteps=1_000_000):
    """
    Trains an SB3 DQN agent on some DeliveryEnv.

    Args:
        project_name (str): The name of the W&B project.
        run_name (str): The name for this specific W&B run.
        total_timesteps (int): The total number of steps to train for.
    """
    # --- Directory Setup -----------------------------------------------
    log_dir = "results"
    videos_dir = os.path.join(log_dir, "videos")
    models_dir = "models"
    tensorboard_dir = os.path.join(log_dir,"tensorboard")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # --- Device Selection ----------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
    # Check for Apple Metal Performance Shaders
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- W&B and Hyperparameter Configuration --------------------------
    # config = {
    #     "policy_type": "MlpPolicy",
    #     "total_timesteps": total_timesteps,
    #     "env_name": "MediumDeliveryEnv",
    #     "map_config": "default",
    #     "learning_rate": 1e-4,
    #     "buffer_size": 100_000,
    #     "learning_starts": 50_000,
    #     "batch_size": 64,
    #     "tau": 1.0,
    #     "gamma": 0.99,
    #     "train_freq": (4, "step"),
    #     "gradient_steps": 1,
    #     "exploration_fraction": 0.5,
    #     "exploration_initial_eps": 1.0,
    #     "exploration_final_eps": 0.05,
    #     "policy_kwargs": dict(net_arch=[256, 256]),
    #     # "seed": SEED,
    #     "device": device
    # }
    config = {
        "policy_type"           : "MlpPolicy",
        "total_timesteps"       : total_timesteps,
        # environment
        "env_name"              : ENV_NAME,
        "map_config"            : MAP_NAME,
        # optimizer
        "learning_rate"         : 5e-5,
        "batch_size"            : 128,
        "gradient_steps"        : 2,
        # replay & targets
        "buffer_size"           : 100_000,
        "learning_starts"       : 75_000,
        "tau"                   : 1.0,
        "gamma"                 : 0.99,
        # frequency
        "train_freq"            : (4, "step"),
        # e-greedy schedule
        "exploration_fraction"  : 0.6,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps" : 0.02,
        # net
        "policy_kwargs"         : dict(net_arch=[256, 256]),
        # misc
        "seed"                  : SEED,
        "device"                : device
    }

    # --- Initialize W&B Run --------------------------------------------
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    
    # --- Environment Setup ---------------------------------------------
    def make_env():
        """Helper function to create and wrap the environment."""

        EnvClass, map_dict = resolve_env()
        map_data = map_dict[MAP_NAME]

        env = EnvClass(map_config=map_data, render_mode='rgb_array')
        if SEED:
            env.seed=SEED

        monitor_path = os.path.join(log_dir, f"monitor_{run.id}") 
        return Monitor(env, filename=monitor_path)

    # Create a vectorized environment
    vec_env = DummyVecEnv([make_env])
    
    # --- Video Recorder Setup ------------------------------------------
    video_record_freq = 50000
    vec_env = VecVideoRecorder(
        vec_env,
        os.path.join(videos_dir, f"{run.id}"),
        record_video_trigger=lambda x: x % video_record_freq == 0,
        video_length=1500
    )

    # --- Model Definition ----------------------------------------------
    model = DQN(
        policy=config['policy_type'],
        env=vec_env,
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
        tensorboard_log=os.path.join(tensorboard_dir, f"tensorboard_{run.id}"),
        device=config['device'],
        seed=config['seed']
    )
    
    # *** FIX: Give the environment a reference to the model ***
    # This allows the environment's render() method to access model.exploration_rate
    for env in vec_env.envs:
        # Need to unwrap the Monitor wrapper to get to the base env
        env.unwrapped.model = model

    # --- Callbacks Setup -----------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // vec_env.num_envs, 1),
        save_path=os.path.join(models_dir, f"{run.id}"),
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=50_000,
        model_save_path=os.path.join(models_dir, f"wandb/{run.id}"),
        model_save_freq=max(50000 // vec_env.num_envs, 1),
        log="all",
        verbose=2,
    )
    
    epsilon_callback = EpsilonLoggerCallback()

    callbacks = [checkpoint_callback, wandb_callback, epsilon_callback]

    # --- Training ------------------------------------------------------
    print("="*50)
    print("Starting training...")
    print("="*50)
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
    finally:
        final_model_path = os.path.join(models_dir, f"dqn_model_final_{run.id}")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        vec_env.close()
        run.finish()
        print("Training complete.")

if __name__ == "__main__":
    train_dqn(run_name=ENV_NAME + "-SB3-DQN", total_timesteps=TOTAL_TIME_STEPS)