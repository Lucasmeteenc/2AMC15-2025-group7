"""
DQN Hyperparameter Sweep Script using Weights & Biases (wandb)
"""
import wandb
from dqn_agent import DQNConfig, train, set_random_seeds

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
            "hidden_dim": {
                "values": [64, 128, 256]
            },
            "memory_size": {
                "values": [50000, 100000, 200000]
            },
            "target_update_interval": {
                "values": [10, 25, 50]
            },
            "epsilon_decay": {
                "values": [0.995, 0.997, 0.999]
            },
            "epsilon_decay": {
                "values": [0.995, 0.997, 0.999]
            },
            "seed": {
                "values": [0, 42, 123]
            },
        }
    }

def sweep_train():
    # Initialize wandb run and get the run object
    wandb_run = wandb.init(project="medium-delivery-dqn-sweep")
    config = wandb.config

    # Build DQNConfig from wandb config
    dqn_config = DQNConfig(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        hidden_dim=config.hidden_dim,
        memory_size=config.memory_size,
        target_update_interval=config.target_update_interval,
        epsilon_decay=config.epsilon_decay,
        seed=config.seed,
    )

    set_random_seeds(dqn_config.seed)
    train(dqn_config, wandb_run=wandb_run)

if __name__ == "__main__":
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="medium-delivery-dqn-sweep")
    wandb.agent(sweep_id, function=sweep_train)