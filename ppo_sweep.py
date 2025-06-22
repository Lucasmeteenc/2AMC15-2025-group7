"""
PPO Hyperparameter Sweep Script using Weights & Biases (wandb)
"""
import wandb
import numpy as np
import torch
from ppo_agent import PPOConfig, PPOAgent, set_random_seeds, make_vec_env

# Define the sweep configuration
def get_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/average_reward",
            "goal": "maximize"
        },
        "parameters": {
            "learning_rate": {
                "values": [1e-4, 3e-4, 1e-3]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "clip_range": {
                "values": [0.1, 0.2, 0.3]
            },
            "gae_lambda": {
                "values": [0.90, 0.95, 0.99]
            },
            "value_coef": {
                "values": [0.25, 0.5, 1.0]
            },
            "entropy_coef": {
                "values": [0.0, 0.01, 0.05]
            },
            "hidden_dim": {
                "values": [32, 64, 128]
            },
            "seed": {
                "values": [0, 42, 123]
            },
        }
    }

def sweep_train():
    # Initialize wandb run and get the run object
    wandb_run = wandb.init(project="medium-delivery-ppo-sweep")
    config = wandb.config

    # Build PPOConfig from wandb config
    ppo_config = PPOConfig(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        clip_range=config.clip_range,
        gae_lambda=config.gae_lambda,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
    )

    set_random_seeds(ppo_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_envs = make_vec_env(ppo_config, ppo_config.num_envs, ppo_config.seed)
    state_space_size = train_envs.single_observation_space.shape[0]
    action_space_size = train_envs.single_action_space.n

    agent = PPOAgent(ppo_config, device, wandb_run)
    agent.setup_networks(state_space_size, action_space_size)
    train_returns = agent.train(train_envs)
    train_envs.close()

    # Final average return
    if train_returns:
        wandb_run.log({"final_avg_return": np.mean(train_returns[-100:])})

if __name__ == "__main__":
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="medium-delivery-ppo-sweep")
    wandb.agent(sweep_id, function=sweep_train)
