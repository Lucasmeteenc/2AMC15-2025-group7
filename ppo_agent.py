"""
Main PPO agent implementation with training loop and configuration.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.data import DataLoader, TensorDataset

from environments.simple_delivery_env import SimpleDeliveryEnv
from ppo_network import NetworkFactory
from ppo_utils import (
    AdvantageCalculator,
    ExplorationScheduler,
    PPOLoss,
    CheckpointManager,
    MetricsLogger,
    PPOError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ppo_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration class for PPO training parameters."""

    hidden_dimensions: int = 512
    num_layers: int = 3
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 128
    ppo_steps: int = 3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    exploration_fraction: float = 0.6
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.1
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.1
    max_grad_norm: float = 0.1
    n_episodes: int = 2000
    learning_starts: int = 100
    reward_threshold: float = 5000.0
    max_steps: int = 10000
    lr_decay: float = 0.995
    checkpoint_interval: int = 500
    use_gae: bool = True
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("Dropout must be between 0 and 1")


class PPOAgent:
    """Main PPO agent class that orchestrates training and evaluation."""

    def __init__(self, config: PPOConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize components
        self.exploration_scheduler = ExplorationScheduler(config)
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.metrics_logger = MetricsLogger(config.log_dir)
        self.advantage_calculator = AdvantageCalculator()

        # Initialize networks (will be set up in setup_networks)
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.actor_scheduler = None
        self.critic_scheduler = None

        logger.info(f"PPO Agent initialized on device: {self.device}")

    def setup_networks(self, state_size: int, action_size: int) -> None:
        """Initialize networks and optimizers."""
        try:
            # Create networks
            self.actor = NetworkFactory.create_actor(
                state_size,
                self.config.hidden_dimensions,
                action_size,
                self.config.dropout,
                self.config.num_layers,
            ).to(self.device)

            self.critic = NetworkFactory.create_critic(
                state_size,
                self.config.hidden_dimensions,
                self.config.dropout,
                self.config.num_layers,
            ).to(self.device)

            # Create optimizers
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=self.config.learning_rate
            )
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.config.learning_rate
            )

            # Create schedulers
            self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer, T_max=self.config.n_episodes, eta_min=1e-6
            )
            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer, T_max=self.config.n_episodes, eta_min=1e-6
            )

            # Log network information
            actor_params = sum(
                p.numel() for p in self.actor.parameters() if p.requires_grad
            )
            critic_params = sum(
                p.numel() for p in self.critic.parameters() if p.requires_grad
            )

            logger.info(f"Actor parameters: {actor_params:,}")
            logger.info(f"Critic parameters: {critic_params:,}")
            logger.info(f"Total parameters: {actor_params + critic_params:,}")

        except Exception as e:
            logger.error(f"Failed to setup networks: {e}")
            raise PPOError(f"Network setup failed: {e}")

    def collect_experience(self, env, episode: int):
        """Collect experience from environment interaction."""
        if self.actor is None or self.critic is None:
            raise PPOError("Networks not initialized. Call setup_networks first.")

        states, actions, log_probs, values, rewards = [], [], [], [], []
        episode_reward = 0.0

        try:
            state, _ = env.reset()
            self.actor.eval()
            self.critic.eval()

            with torch.no_grad():
                done = False
                for step in range(self.config.max_steps):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                    # --- 1. Forward pass -------------------------------------------------
                    action_logits = self.actor(state_tensor)
                    action_probs = F.softmax(action_logits, dim=-1)
                    dist = distributions.Categorical(action_probs)
                    value = self.critic(state_tensor)

                    # --- 2. ε-greedy exploration ----------------------------------------
                    eps = 0
                    if np.random.rand() < eps:
                        weights = self.exploration_scheduler.get_exploration_weights(
                            state, env, episode
                        )
                        a = np.random.choice(len(weights), p=weights)
                        log_p = torch.clamp(
                            torch.log(
                                torch.tensor(weights[a] + 1e-8, device=self.device)
                            ),
                            min=-10,
                            max=2,
                        ).unsqueeze(0)
                    else:
                        a = dist.sample().item()
                        log_p = dist.log_prob(
                            torch.tensor(a, device=self.device)
                        ).unsqueeze(0)

                    next_state, reward, terminated, truncated, _ = env.step(a)
                    done = terminated or truncated

                    # store transition
                    states.append(state_tensor.squeeze(0))
                    actions.append(torch.tensor([a], device=self.device))
                    log_probs.append(log_p.squeeze() if log_p.dim() > 0 else log_p)
                    values.append(value.squeeze())
                    rewards.append(reward)
                    episode_reward += reward
                    state = next_state

                    if done:
                        break

                # Bootstrap value for truncated episodes
                if not done:
                    final_state_tensor = (
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    )
                    final_value = self.critic(final_state_tensor).squeeze().item()
                else:
                    final_value = 0.0

            # Convert to tensors
            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            log_probs = torch.stack(log_probs).to(self.device)
            values = torch.stack(values).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            return (
                episode_reward,
                states,
                actions,
                log_probs,
                values,
                rewards,
                final_value,
            )

        except Exception as e:
            logger.error(f"Error during experience collection: {e}")
            raise PPOError(f"Experience collection failed: {e}")

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Update policy using PPO algorithm."""
        if None in [
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
        ]:
            raise PPOError("Networks or optimizers not initialized")

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        try:
            # Detach old values to prevent gradients
            log_probs_old = log_probs_old.detach()
            advantages = advantages.detach()
            returns = returns.detach()

            # Create dataset and dataloader
            dataset = TensorDataset(states, actions, log_probs_old, advantages, returns)
            dataloader = DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=True
            )

            self.actor.train()
            self.critic.train()

            for _ in range(self.config.ppo_steps):
                for batch_data in dataloader:
                    (
                        states_batch,
                        actions_batch,
                        log_probs_old_batch,
                        advantages_batch,
                        returns_batch,
                    ) = batch_data

                    # Forward pass through both networks
                    action_logits = self.actor(states_batch)
                    values = self.critic(states_batch).squeeze(-1)
                    values = torch.clamp(values, -100, 100)

                    # Calculate new log probabilities
                    action_probs = F.softmax(action_logits, dim=-1)
                    action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
                    dist = distributions.Categorical(action_probs)
                    log_probs_new = dist.log_prob(actions_batch)
                    log_probs_new = torch.clamp(log_probs_new, min=-10, max=2)
                    log_probs_old_batch = torch.clamp(
                        log_probs_old_batch, min=-10, max=2
                    )

                    # Calculate losses
                    policy_loss = PPOLoss.calculate_policy_loss(
                        log_probs_old_batch,
                        log_probs_new,
                        returns_batch,
                        self.config.epsilon,
                    )

                    value_loss = F.mse_loss(values, returns_batch.clamp(-50, 50))
                    entropy = dist.entropy().mean()
                    entropy_loss = -self.config.entropy_coefficient * entropy

                    # Update actor
                    actor_loss = policy_loss + entropy_loss
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()

                    total_norm = 0
                    for p in self.actor.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5

                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.config.max_grad_norm
                    )
                    self.actor_optimizer.step()

                    # Update critic
                    critic_loss = self.config.value_coefficient * value_loss
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.config.max_grad_norm
                    )
                    self.critic_optimizer.step()

                    # print(f"log_probs_old: {log_probs_old_batch[:5]}")
                    # print(f"log_probs_new: {log_probs_new[:5]}")
                    # print(f"advantages: {advantages_batch[:5]}")
                    # print(f"ratio: {torch.exp(log_probs_new[:5] - log_probs_old_batch[:5])}")

                    # Track losses
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    print(f"Action logits range: [{action_logits.min():.3f}, {action_logits.max():.3f}]")

            # Calculate average losses
            num_updates = self.config.ppo_steps * len(dataloader)
            avg_policy_loss = total_policy_loss / num_updates
            avg_value_loss = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates

            return avg_policy_loss, avg_value_loss, avg_entropy

        except Exception as e:
            logger.error(f"Error during policy update: {e}")
            raise PPOError(f"Policy update failed: {e}")

    def evaluate(self, env) -> float:
        """Evaluate the current policy."""
        if self.actor is None or self.critic is None:
            raise PPOError("Networks not initialized")

        self.actor.eval()
        self.critic.eval()
        episode_reward = 0
        state, _ = env.reset()

        with torch.no_grad():
            for step in range(self.config.max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_logits = self.actor(state_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                episode_reward += reward
                state = next_state

                if done:
                    break

        return episode_reward

    def train(self, train_env, eval_env):
        """Main training loop."""
        train_rewards = []
        policy_losses = []
        value_losses = []
        entropies = []

        logger.info(f"Starting PPO training for {self.config.n_episodes} episodes")

        try:
            for episode in range(1, self.config.n_episodes + 1):
                # Collect experience
                (
                    episode_reward,
                    states,
                    actions,
                    log_probs,
                    values,
                    rewards,
                    final_value,
                ) = self.collect_experience(train_env, episode)

                # Calculate returns and advantages
                returns = self.advantage_calculator.calculate_returns(
                    rewards, self.config.discount_factor, final_value
                )

                if self.config.use_gae:
                    advantages = self.advantage_calculator.calculate_gae(
                        rewards,
                        values,
                        self.config.discount_factor,
                        self.config.gae_lambda,
                        final_value,
                    )
                else:
                    advantages = self.advantage_calculator.calculate_simple_advantages(
                        returns, values
                    )

                # Update policy
                avg_policy_loss, avg_value_loss, avg_entropy = self.update_policy(
                    states, actions, log_probs, advantages, returns
                )

                # Step learning rate schedulers
                if self.actor_scheduler and self.critic_scheduler:
                    self.actor_scheduler.step()
                    self.critic_scheduler.step()

                # Track metrics
                train_rewards.append(episode_reward)
                policy_losses.append(avg_policy_loss)
                value_losses.append(avg_value_loss)
                entropies.append(avg_entropy)

                # Logging
                if episode % 100 == 0:
                    avg_reward = np.mean(train_rewards[-50:])
                    current_actor_lr = (
                        self.actor_scheduler.get_last_lr()[0]
                        if self.actor_scheduler
                        else self.config.learning_rate
                    )
                    current_critic_lr = (
                        self.critic_scheduler.get_last_lr()[0]
                        if self.critic_scheduler
                        else self.config.learning_rate
                    )

                    logger.info(
                        f"Ep: {episode:4d} | Avg Reward: {avg_reward:.2f} | "
                        f"Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f} | "
                        f"Entropy: {avg_entropy:.4f} | Actor LR: {current_actor_lr:.6f} | "
                        f"Critic LR: {current_critic_lr:.6f}"
                    )

                    self.metrics_logger.log_training_metrics(
                        episode,
                        avg_reward,
                        avg_policy_loss,
                        avg_value_loss,
                        avg_entropy,
                        current_actor_lr,
                        current_critic_lr,
                    )

                # Evaluation
                if episode % 500 == 0:
                    eval_reward = self.evaluate(eval_env)
                    logger.info(f"Evaluation Reward: {eval_reward:.2f}")
                    self.metrics_logger.log_evaluation_metrics(episode, eval_reward)

                # Checkpointing
                if episode % self.config.checkpoint_interval == 0:
                    checkpoint_path = f"checkpoint_ep{episode}.pt"
                    self.checkpoint_manager.save_checkpoint(
                        self.actor,
                        self.critic,
                        self.actor_optimizer,
                        self.critic_optimizer,
                        episode,
                        train_rewards,
                        self.config,
                        checkpoint_path,
                    )

                # Early stopping
                if (
                    len(train_rewards) >= 100
                    and np.mean(train_rewards[-100:]) >= self.config.reward_threshold
                ):
                    logger.info(
                        f"Reached reward threshold! Training stopped at episode {episode}"
                    )
                    break

            # Save final model
            self.checkpoint_manager.save_checkpoint(
                self.actor,
                self.critic,
                self.actor_optimizer,
                self.critic_optimizer,
                episode,
                train_rewards,
                self.config,
                "final_model.pt",
            )

            return train_rewards

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise PPOError(f"Training loop failed: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with PPOConfig defaults."""
    parser = argparse.ArgumentParser(description="PPO Training Configuration")

    # Get default config for reference
    default_config = PPOConfig()

    # Network architecture
    parser.add_argument(
        "--hidden_dimensions", type=int, default=default_config.hidden_dimensions
    )
    parser.add_argument("--num_layers", type=int, default=default_config.num_layers)
    parser.add_argument("--dropout", type=float, default=default_config.dropout)

    # Learning parameters
    parser.add_argument(
        "--learning_rate", type=float, default=default_config.learning_rate
    )
    parser.add_argument("--batch_size", type=int, default=default_config.batch_size)
    parser.add_argument("--n_episodes", type=int, default=default_config.n_episodes)
    parser.add_argument("--ppo_steps", type=int, default=default_config.ppo_steps)

    # PPO parameters
    parser.add_argument(
        "--discount_factor", type=float, default=default_config.discount_factor
    )
    parser.add_argument("--epsilon", type=float, default=default_config.epsilon)
    parser.add_argument(
        "--use_gae", action="store_true", default=default_config.use_gae
    )

    # Directories
    parser.add_argument("--log_dir", type=str, default=default_config.log_dir)
    parser.add_argument(
        "--checkpoint_dir", type=str, default=default_config.checkpoint_dir
    )

    # Additional
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)

    return parser


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_environment(render_mode: Optional[str] = None):
    """Create and validate environment."""
    try:
        env = SimpleDeliveryEnv(render_mode=render_mode)
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        raise PPOError(f"Environment creation failed: {e}")


def main():
    """Main training function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    config = PPOConfig()
    set_random_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create environments
    train_env = create_environment(render_mode=None)
    eval_env = create_environment(render_mode="human")

    # Get environment dimensions
    state_space_size = train_env.observation_space.shape[0]
    action_space_size = train_env.action_space.n

    # Create and setup PPO agent
    agent = PPOAgent(config, device)
    agent.setup_networks(state_space_size, action_space_size)

    # Start training
    train_rewards = agent.train(train_env, eval_env)

    logger.info("Training completed successfully!")
    logger.info(f"Final average reward: {np.mean(train_rewards[-100:]):.2f}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
