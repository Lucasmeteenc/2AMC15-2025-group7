"""
Utility classes for PPO implementation.
Contains advantage calculation, checkpointing, and logging.
"""

import csv
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class PPOError(Exception):
    """Custom exception for PPO-related errors."""
    pass


class AdvantageCalculator:
    """Utility class for calculating advantages and returns."""

    @staticmethod
    def calculate_returns(
        rewards: torch.Tensor,  # (T, N)
        dones: torch.Tensor,  # (T, N) bool
        last_values: torch.Tensor,  # (N,)
        gamma: float,
    ) -> torch.Tensor:
        """
        Discounted returns without GAE (equivalent to λ = 1).
        """
        T, N = rewards.shape
        device = rewards.device

        returns = torch.zeros((T, N), device=device)
        running = last_values  # V(s_T)

        for t in reversed(range(T)):
            mask = 1.0 - dones[t].float()  # zero after terminal
            running = rewards[t] + gamma * running * mask
            returns[t] = running
        return returns

    @staticmethod
    def calculate_gae(
        rewards: torch.Tensor,  # (T, N)
        values: torch.Tensor,  # (T, N)
        dones: torch.Tensor,  # (T, N)  bool
        last_values: torch.Tensor,  # (N,)
        gamma: float,
        lam: float,
    ):
        """Returns (advantages, returns) each shaped (T, N)."""
        T, N = rewards.shape
        device = rewards.device

        # Append the bootstrap value so we can vectorise index t+1
        values_ext = torch.cat([values, last_values.unsqueeze(0)], dim=0)  # (T+1, N)

        advantages = torch.zeros((T, N), device=device)
        gae = torch.zeros(N, device=device)

        for t in reversed(range(T)):
            # If episode ended at step t, mask = 0 → cut trace
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * values_ext[t + 1] * mask - values[t]
            gae = delta + gamma * lam * gae * mask
            advantages[t] = gae

        returns = advantages + values  # V-targets
        return advantages, returns
    
    @staticmethod
    def flatten_time_env(tensor: torch.Tensor) -> torch.Tensor:
        """Collapse (time, env) into batch dimension."""
        return tensor.reshape(-1, *tensor.shape[2:])


class CheckpointManager:
    """Manages model checkpoints and state persistence."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        episode: int,
        rewards: List[float],
        config,
        filepath: Optional[str] = None,
    ) -> str:
        """Save model checkpoint."""
        if filepath is None:
            filepath = self.checkpoint_dir / f"checkpoint_ep{episode}.pt"

        try:
            checkpoint_data = {
                "episode": episode,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "actor_optimizer_state_dict": actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": critic_optimizer.state_dict(),
                "rewards": rewards,
                "config": config,
                "pytorch_version": torch.__version__,
            }

            torch.save(checkpoint_data, filepath)
            logger.info(f"Checkpoint saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise PPOError(f"Checkpoint saving failed: {e}")

    def load_checkpoint(
        self,
        filepath: str,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: Optional[optim.Optimizer] = None,
        critic_optimizer: Optional[optim.Optimizer] = None,
    ):
        """Load model checkpoint."""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

            checkpoint = torch.load(filepath, map_location="cpu")

            actor.load_state_dict(checkpoint["actor_state_dict"])
            critic.load_state_dict(checkpoint["critic_state_dict"])

            if actor_optimizer is not None:
                actor_optimizer.load_state_dict(
                    checkpoint["actor_optimizer_state_dict"]
                )
            if critic_optimizer is not None:
                critic_optimizer.load_state_dict(
                    checkpoint["critic_optimizer_state_dict"]
                )

            logger.info(f"Checkpoint loaded: {filepath}")

            return (
                checkpoint["episode"],
                checkpoint["rewards"],
                checkpoint.get("config", None),
            )

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise PPOError(f"Checkpoint loading failed: {e}")


class MetricsLogger:
    """Handles logging and CSV output for training metrics."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_log_file = self.log_dir / "training_log.csv"
        self.eval_log_file = self.log_dir / "evaluation_log.csv"
        self._initialize_csv_files()

    def _initialize_csv_files(self) -> None:
        """Initialize CSV log files with headers."""
        try:
            with open(self.train_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "episode",
                        "average_reward",
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "actor_lr",
                        "critic_lr",
                    ]
                )

            with open(self.eval_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "evaluation_reward"])

        except Exception as e:
            logger.error(f"Failed to initialize CSV files: {e}")
            raise PPOError(f"CSV initialization failed: {e}")

    def log_training_metrics(
        self,
        episode: int,
        avg_reward: float,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        actor_lr: float,
        critic_lr: float,
    ) -> None:
        """Log training metrics to CSV."""
        try:
            with open(self.train_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        episode,
                        avg_reward,
                        policy_loss,
                        value_loss,
                        entropy,
                        actor_lr,
                        critic_lr,
                    ]
                )
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")

    def log_evaluation_metrics(self, episode: int, eval_reward: float) -> None:
        """Log evaluation metrics to CSV."""
        try:
            with open(self.eval_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode, eval_reward])
        except Exception as e:
            logger.error(f"Failed to log evaluation metrics: {e}")
