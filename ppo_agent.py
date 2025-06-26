"""
Main PPO agent implementation with training loop and configuration.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import wandb

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from environments.medium_delivery_env import MediumDeliveryEnv
from ppo_network import NetworkFactory
from ppo_utils import (
    AdvantageCalculator,
    CheckpointManager,
    MetricsLogger,
    PPOError,
)

import gymnasium as gym
from maps import MAIL_DELIVERY_MAPS

EVAL_FREQUENCY_STEPS = 20_000  # Frequency of evaluation in training steps

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

    # Sampler
    total_timesteps: int = 2_000_000  # stop criterion
    horizon: int = 2_048  # T  (steps per env before an update)
    num_envs: int = 8  # N  (parallel envs)

    # Architecture
    hidden_dim: int = 128

    # Optimization
    learning_rate: float = 1e-3  # Adam
    batch_size: int = 64  # per SGD minibatch
    n_epochs: int = 10  # PPO epochs per update

    clip_range: float = 0.1  # ε in clipped objective
    gamma: float = 0.99  # discount
    gae_lambda: float = 0.9  # λ for GAE
    value_coef: float = 1  # c1
    entropy_coef: float = 0.0  # c2
    max_grad_norm: float = 0.5  # global grad-clip

    use_gae: bool = True

    # Rest
    seed: int = 0
    log_interval: int = 1  # updates between train logs
    checkpoint_interval: int = 1  # updates between checkpoints
    patience: int = (
        10  # Number of consecutive windows without improvement before stopping
    )
    log_window: int = 100

    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints_ppo"
    video_dir: str = "videos"
    map_name: str = "default"

    def __post_init__(self):
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if self.batch_size < 1 or self.batch_size > self.horizon * self.num_envs:
            raise ValueError("batch_size must be in (0, horizon * num_envs]")
        if not (0.0 < self.learning_rate):
            raise ValueError("learning_rate must be positive")


class PPOAgent:
    """Main PPO agent class that orchestrates training and evaluation."""

    def __init__(
        self, config: PPOConfig, device: Optional[torch.device] = None, wandb_run=None
    ):
        self.config = config
        self.wandb = wandb_run
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize components
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.metrics_logger = MetricsLogger(config.log_dir)
        self.advantage_calculator = AdvantageCalculator()

        # Initialize best average return
        self.best_avg_return = float("-inf")
        # Counter for epochs without improvement
        self.epochs_without_improvement = 0

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
                action_size,
            ).to(self.device)

            self.critic = NetworkFactory.create_critic(
                state_size,
            ).to(self.device)

            # Create optimizers
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=self.config.learning_rate
            )
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.config.learning_rate
            )

            # Create schedulers
            # self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            #     self.actor_optimizer, T_max=self.config.n_episodes, eta_min=1e-6
            # )
            # self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            #     self.critic_optimizer, T_max=self.config.n_episodes, eta_min=1e-6
            # )

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

    def collect_rollout(self, envs, obs):
        """
        Collect `self.horizon` steps from `n_envs` vectorised environments.
        Returns a dict ready for GAE and PPO updates.
        """
        storage = []

        for _ in range(self.config.horizon):
            # 1. Forward pass
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()  # (N,)

            next_obs_np, rewards_np, term_np, trunc_np, _ = envs.step(
                actions.cpu().numpy()
            )
            done_np = np.logical_or(term_np, trunc_np)  # (N,)

            # 2. Step environments
            next_obs = torch.from_numpy(next_obs_np).to(self.device)
            rewards = torch.from_numpy(rewards_np).to(self.device)
            dones = torch.from_numpy(done_np.astype(np.uint8)).to(self.device)

            storage.append(
                {
                    "obs": obs,
                    "act": actions,
                    "logp": dist.log_prob(actions),
                    "value": self.critic(obs).squeeze(-1),
                    "reward": rewards,
                    "done": dones,
                }
            )

            # 3. Reset envs that ended and continue
            obs = next_obs

        last_values = self.critic(obs).squeeze(-1)  # (N,)
        return storage, last_values, obs

    def update_policy(
        self,
        states: torch.Tensor,  # (B, obs_dim)
        actions: torch.Tensor,  # (B,)
        logp_old: torch.Tensor,  # (B,)
        advantages: torch.Tensor,  # (B,)
        returns: torch.Tensor,  # (B,)
    ) -> Tuple[float, float, float]:
        """
        One PPO update consisting of `self.config.n_epochs` epochs of SGD
        on minibatches of size `self.config.batch_size`.
        Expects *flattened* rollout tensors (B = T × N).
        """
        if any(
            x is None
            for x in (
                self.actor,
                self.critic,
                self.actor_optimizer,
                self.critic_optimizer,
            )
        ):
            raise PPOError("Networks or optimizers not initialized")

        # Advantage normalisation (recommended in PPO paper appendix)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(states, actions, logp_old, advantages, returns)
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True
        )

        self.actor.train()
        self.critic.train()

        tot_pi_loss = tot_v_loss = tot_entropy = 0.0
        n_minibatch = self.config.n_epochs * len(loader)
        clip_eps = self.config.clip_range

        for _ in range(self.config.n_epochs):
            for s_b, a_b, logp_old_b, adv_b, ret_b in loader:
                # detach minibatch views
                s_b = s_b.detach()
                a_b = a_b.detach()
                logp_old_b = logp_old_b.detach()
                adv_b = adv_b.detach()
                ret_b = ret_b.detach()

                # forwardpass actor
                logits = self.actor(s_b)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a_b)
                entropy = dist.entropy().mean()  # entropy bonus for exploration

                # clipped surrogate loss
                ratio = torch.exp(logp - logp_old_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss = policy_loss - self.config.entropy_coef * entropy
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()

                # Forwardpass Critic
                values = self.critic(s_b).squeeze(-1)
                value_loss = F.mse_loss(values, ret_b)  # value function loss

                self.critic_optimizer.zero_grad()
                (self.config.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                self.critic_optimizer.step()

                # Logging
                tot_pi_loss += policy_loss.item()
                tot_v_loss += value_loss.item()
                tot_entropy += entropy.item()

        return (
            tot_pi_loss / n_minibatch,
            tot_v_loss / n_minibatch,
            tot_entropy / n_minibatch,
        )

    @torch.no_grad()
    def evaluate(self, env: gym.Env) -> float:
        """
        Roll one evaluation episode.

        Parameters
        ----------
        env    : a *single* Gymnasium environment

        Returns
        -------
        episode_return : float
        """
        obs_np, _ = env.reset()
        obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(self.device)

        episode_return = 0.0
        done = False
        while not done:
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            next_obs_np, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward

            obs = torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32)).to(
                self.device
            )

        return episode_return

    def train(self, train_envs, save_update=True):
        """
        Main PPO training loop (vectorised environments, fixed-horizon roll-outs).
        - envs : vectorised training environments (n_envs)
        - save_update : whether to save the model after each update
        """
        horizon = self.config.horizon
        n_envs = self.config.num_envs
        total_steps = self.config.total_timesteps
        n_updates = total_steps // (horizon * n_envs)

        # logging
        running_return = np.zeros(self.config.num_envs, dtype=np.float32)
        completed_returns: list[float] = []

        obs, _ = train_envs.reset()
        obs = torch.from_numpy(obs).to(self.device)

        total_env_training_steps = 0
        next_evaluation_at_step = EVAL_FREQUENCY_STEPS

        eval_env = create_environment(self.config, seed=self.config.seed + 1)

        for update in range(1, n_updates + 1):
            # 1. Roll-out
            storage, last_values, obs = self.collect_rollout(train_envs, obs)

            # unpack into T × n_envs tensors
            rewards = torch.stack([b["reward"] for b in storage])  # (T, n_envs)
            dones = torch.stack([b["done"] for b in storage])  # (T, n_envs)
            values = torch.stack([b["value"] for b in storage])  # (T, n_envs)
            logp = torch.stack([b["logp"] for b in storage])
            actions = torch.stack([b["act"] for b in storage])
            states = torch.stack([b["obs"] for b in storage])

            # Quick fix for mismatching dimensions in 1d
            if n_envs == 1:
                if rewards.dim() == 1:
                    rewards = rewards.unsqueeze(1)
                if dones.dim() == 1:
                    dones = dones.unsqueeze(1)
                if values.dim() == 1:
                    values = values.unsqueeze(1)
                if logp.dim() == 1:
                    logp = logp.unsqueeze(1)
                if actions.dim() == 1:
                    actions = actions.unsqueeze(1)
                if states.dim() == 2:
                    states = states.unsqueeze(1)
                if last_values.dim() == 0:
                    last_values = last_values.unsqueeze(0)

            # 2. Advantage / return calculation
            if self.config.use_gae:
                advantages, returns = self.advantage_calculator.calculate_gae(
                    rewards,
                    values,
                    dones,
                    last_values,
                    gamma=self.config.gamma,
                    lam=self.config.gae_lambda,
                )
            else:
                returns = self.advantage_calculator.calculate_returns(
                    rewards,
                    dones,
                    last_values,
                    gamma=self.config.gamma,
                )
                advantages = returns - values

            # flatten to (T*n_envs, …) for minibatching
            states_f = AdvantageCalculator.flatten_time_env(states)
            actions_f = AdvantageCalculator.flatten_time_env(actions)
            logp_f = AdvantageCalculator.flatten_time_env(logp)
            returns_f = AdvantageCalculator.flatten_time_env(returns)
            adv_f = AdvantageCalculator.flatten_time_env(advantages)

            # 3. PPO policy & value update
            pi_loss, v_loss, entropy = self.update_policy(
                states_f, actions_f, logp_f, adv_f, returns_f
            )

            # 4. LR schedulers
            if self.actor_scheduler:
                self.actor_scheduler.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            # 5. Episode-level reward bookkeeping
            # accumulate rewards per env and extract completed returns
            rew_np = rewards.cpu().numpy()
            done_np = dones.cpu().numpy()
            for env_i in range(self.config.num_envs):
                for r, d in zip(rew_np[:, env_i], done_np[:, env_i]):
                    running_return[env_i] += r
                    if d:
                        completed_returns.append(float(running_return[env_i]))
                        running_return[env_i] = 0.0

            # 6. Logging
            if update % self.config.log_interval == 0:
                recent = completed_returns[-self.config.log_window :]
                avg_ret = np.mean(recent) if recent else 0.0
                logger.info(
                    f"Upd {update:4d}/{n_updates} | "
                    f"AvgEpRet {avg_ret:8.3f} | "
                    f"PiLoss {pi_loss:8.4f} | "
                    f"VLoss {v_loss:8.4f} | "
                    f"Entropy {entropy:6.3f}"
                )
                self.metrics_logger.log_training_metrics(
                    update,
                    avg_ret,
                    pi_loss,
                    v_loss,
                    entropy,
                    self.actor_optimizer.param_groups[0]["lr"],
                    self.critic_optimizer.param_groups[0]["lr"],
                )
            if self.wandb and update % self.config.log_interval == 0:
                steps_per_update = self.config.horizon * self.config.num_envs
                self.wandb.log(
                    {
                        "train/avg_ep_return": avg_ret,
                        "train/policy_loss": pi_loss,
                        "train/value_loss": v_loss,
                        "train/entropy": entropy,
                        "train/actor_lr": self.actor_optimizer.param_groups[0]["lr"],
                        "train/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                        "global_step": update * steps_per_update,
                    }
                )

            # 7. Evaluation
            # Increment total_env_training_steps
            steps_this_update = self.config.horizon * self.config.num_envs
            total_env_training_steps += steps_this_update
            # Evaluation at fixed step intervals
            if self.wandb and total_env_training_steps >= next_evaluation_at_step:
                total_score = 0.0
                runs = 10
                for _ in range(runs):
                    # Evaluate multiple times to get a more stable average
                    eval_ret = self.evaluate(eval_env)
                    total_score += eval_ret
                eval_ret = total_score / runs
                self.wandb.log(
                    {
                        "eval/average_reward": eval_ret,
                        "eval/total_training_steps": total_env_training_steps,
                    }
                )
                next_evaluation_at_step += EVAL_FREQUENCY_STEPS

            # 8. Checkpoint
            run_id = (
                self.wandb.id
                if self.wandb and hasattr(self.wandb, "id")
                else "no_wandb"
            )
            if save_update and update % self.config.checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.actor,
                    self.critic,
                    self.actor_optimizer,
                    self.critic_optimizer,
                    update,
                    completed_returns,
                    self.config,
                    f"ckpt_update{update}_{run_id}.pth",
                )

            # 9. Early Stopping
            if update % self.config.log_interval == 0:
                if avg_ret > self.best_avg_return:
                    self.best_avg_return = avg_ret  # Update best average return
                    self.epochs_without_improvement = 0  # Reset counter
                else:
                    self.epochs_without_improvement += 1  # Increment counter

                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(
                        f"Stopping early at update {update} due to no improvement in average return for {self.config.patience} consecutive epochs."
                    )
                    break

        # final save
        final_model_filename = f"final_model_{run_id}.pth"
        self.checkpoint_manager.save_checkpoint(
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
            update,
            completed_returns,
            self.config,
            final_model_filename,
        )
        return completed_returns


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with PPOConfig defaults."""
    p = argparse.ArgumentParser(description="PPO Training Configuration")
    default = PPOConfig()

    # Sampler
    p.add_argument(
        "--num-envs",
        type=int,
        default=default.num_envs,
        help="Number of parallel environments (N)",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=default.horizon,
        help="Roll-out length per env before each optimisation phase (T)",
    )
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=default.total_timesteps,
        help="Stop training after this many environment steps",
    )

    # Network
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=default.hidden_dim,
        help="Width of each of the two fully-connected layers",
    )

    # Optimization
    p.add_argument("--learning-rate", type=float, default=default.learning_rate)
    p.add_argument(
        "--batch-size",
        type=int,
        default=default.batch_size,
        help="Minibatch size for SGD inside each PPO update",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=default.n_epochs,
        help="Number of gradient passes per PPO update",
    )
    p.add_argument(
        "--clip-range",
        type=float,
        default=default.clip_range,
        help="ε in the PPO clipped objective",
    )
    p.add_argument("--gamma", type=float, default=default.gamma)
    p.add_argument("--gae-lambda", type=float, default=default.gae_lambda)
    p.add_argument("--value-coef", type=float, default=default.value_coef)
    p.add_argument("--entropy-coef", type=float, default=default.entropy_coef)
    p.add_argument("--max-grad-norm", type=float, default=default.max_grad_norm)
    p.add_argument("--use-gae", type=bool, default=default.use_gae)

    # Logging & checkpointing
    p.add_argument("--seed", type=int, default=default.seed)
    p.add_argument("--log-interval", type=int, default=default.log_interval)
    p.add_argument(
        "--checkpoint-interval", type=int, default=default.checkpoint_interval
    )
    p.add_argument("--log-window", type=int, default=default.log_window)

    # Directories
    p.add_argument("--log-dir", type=str, default=default.log_dir)
    p.add_argument("--checkpoint-dir", type=str, default=default.checkpoint_dir)
    p.add_argument("--video-dir", type=str, default=default.video_dir)
    p.add_argument("--map-name", type=str, default=default.map_name)

    return p


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_environment(config: PPOConfig, seed: int, render_mode: Optional[str] = None):
    """Create and validate environment."""
    try:
        env = MediumDeliveryEnv(
            map_config=MAIL_DELIVERY_MAPS[config.map_name],
            render_mode=render_mode,
            seed=seed,
        )
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        raise PPOError(f"Environment creation failed: {e}")


def make_vec_env(config: PPOConfig, num_envs: int, seed: int) -> gym.vector.VectorEnv:
    """Return a SyncVectorEnv (1) or AsyncVectorEnv (≥2)."""

    def _factory(rank: int):
        def _thunk():
            env = create_environment(config, seed=config.seed, render_mode=None)
            env.action_space.seed(seed + rank)
            return env

        return _thunk

    env_fns = [_factory(i) for i in range(num_envs)]
    if num_envs == 1:
        return gym.vector.SyncVectorEnv(env_fns)
    return gym.vector.AsyncVectorEnv(env_fns)


def main() -> None:
    """
    Launch PPO training with vectorised training envs
    and a single (renderable) evaluation env.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    config = PPOConfig(**vars(args))

    wandb_run = wandb.init(
        project="medium-inside-delivery-ppo",
        name=f"PPO-{config.num_envs}envs-{config.horizon}T",
        config=config.__dict__,
        tags=["ppo", "gymnasium"],
    )

    set_random_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Vectorised training environments
    train_envs = make_vec_env(config, config.num_envs, config.seed)

    # 2. Dimensionality from ONE env instance
    state_space_size = train_envs.single_observation_space.shape[0]
    action_space_size = train_envs.single_action_space.n

    # 3. PPO Agent
    agent = PPOAgent(config, device, wandb_run)
    agent.setup_networks(state_space_size, action_space_size)

    # 4. Training loop
    train_returns = agent.train(train_envs)

    # 5. Log final results
    logger.info("Training completed successfully!")
    if train_returns:
        logger.info(
            f"Final 100-episode avg return: {np.mean(train_returns[-100:]):.2f}"
        )

    wandb_run.finish()
    train_envs.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
