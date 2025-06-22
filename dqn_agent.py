"""
Main DQN agent implementation with training loop and configuration.
Exponential decay. 2 hidden layers.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import random
from collections import deque
import wandb

from model_custom_dqn import Linear_QNetGen2, QTrainer
from environments.medium_delivery_env import MediumDeliveryEnv
from maps import MAIL_DELIVERY_MAPS

EVAL_FREQUENCY_STEPS = 20_000  # Frequency of evaluation in training steps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dqn_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

@dataclass
class DQNConfig:
    total_episodes: int = 15_000
    batch_size: int = 32
    memory_size: int = 100_000          # Increased from 50_000
    learning_rate: float = 0.0001       # Decreased from 0.0005
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.997
    hidden_dim: int = 256               # Increased from 64
    checkpoint_dir: str = "checkpoints_dqn"
    log_interval: int = 10
    seed: int = 0
    map_name: str = "default"
    target_update_interval: int = 25
    lr_decay_factor: float = 0.5        # Factor to multiply LR by
    lr_decay_episodes: int = 5000       # How often to decay LR (in episodes)
    min_learning_rate: float = 0.00001  # Minimum learning rate
    checkpoint_interval: int = 2500     # Updates between checkpoints

class DQNAgent:
    def __init__(self, observation_space, action_space, config: DQNConfig):
        self.n_game = 0
        self.epsilon = config.epsilon_start
        self.gamma = config.gamma
        self.memory = deque(maxlen=config.memory_size)
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_size = action_space.n
        self.observation_size = observation_space.shape[0]
        self.model = Linear_QNetGen2(self.observation_size, config.hidden_dim, self.action_size)
        self.target_model = Linear_QNetGen2(self.observation_size, config.hidden_dim, self.action_size)  # Target network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(self.model, lr=config.learning_rate, gamma=self.gamma)
        self.config = config

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.config.batch_size:
            mini_sample = random.sample(self.memory, self.config.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, done = zip(*mini_sample)
        batch_loss = self._train_step_with_target(states, actions, rewards, next_states, done)
        return batch_loss

    def train_short_memory(self, state, action, reward, next_state, done):
        _ = self._train_step_with_target([state], [action], [reward], [next_state], [done])

    def _train_step_with_target(self, states, actions, rewards, next_states, done):
        # Use target network for next_state Q-values
        state = torch.tensor(np.array(states), dtype=torch.float)
        next_state = torch.tensor(np.array(next_states), dtype=torch.float)
        action = torch.tensor(np.array(actions), dtype=torch.long)
        reward = torch.tensor(np.array(rewards), dtype=torch.float)
        done = torch.tensor(np.array(done), dtype=torch.bool)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        pred = self.model(state)
        target = pred.clone().detach()
        next_q = self.target_model(next_state).detach()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(next_q[idx])
            target[idx][action[idx]] = Q_new
        self.trainer.optimizer.zero_grad()
        loss = self.trainer.criterion(target, pred)
        loss.backward()
        self.trainer.optimizer.step()
        return loss.item()

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            return torch.argmax(prediction).item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name, run_id=None):
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True)
        if run_id:
            file_name = f"{self.config.checkpoint_dir}/model_{name}_{run_id}.pth"
        else:
            file_name = f"{self.config.checkpoint_dir}/model_{name}.pth"
        self.model.save(file_name=file_name)
        logger.info(f"Model saved to {file_name}")

def set_random_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def create_environment(config: DQNConfig, seed=None):
    if seed is not None:
        env = MediumDeliveryEnv(map_config=MAIL_DELIVERY_MAPS[config.map_name], render_mode=None, seed=seed)
    else:
        env = MediumDeliveryEnv(map_config=MAIL_DELIVERY_MAPS[config.map_name], render_mode=None)
    return env

def evaluate_policy(agent, env, config: DQNConfig, n_episodes: int = 3):
    """Evaluate agent with epsilon=0 (greedy policy) and return average score."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    total_score = 0.0
    for _ in range(n_episodes):
        state, info = env.reset()
        done = False
        score = 0.0
        while not done:
            action = agent.get_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
            if terminated or truncated:
                break
        total_score += score
    agent.epsilon = original_epsilon
    return total_score / n_episodes

def train(config: DQNConfig, wandb_run=None):
    set_random_seeds(config.seed)
    env = create_environment(config, seed=config.seed)
    eval_env = create_environment(config, seed=config.seed + 1)
    agent = DQNAgent(env.observation_space, env.action_space, config)
    state_old, _ = env.reset()
    cummulative_reward = 0
    total_env_training_steps = 0
    next_evaluation_at_step = EVAL_FREQUENCY_STEPS
    run_id = None
    if wandb_run is not None and hasattr(wandb_run, 'id'):
        run_id = wandb_run.id
    for episode in range(1, config.total_episodes + 1):
        state_old, _ = env.reset()
        cummulative_reward = 0
        done = False
        while not done:
            action = agent.get_action(state_old)
            state_new, reward, terminated, truncated, _ = env.step(action)
            total_env_training_steps += 1
            cummulative_reward += reward
            agent.train_short_memory(state_old, action, reward, state_new, terminated)
            agent.remember(state_old, action, reward, state_new, terminated)
            state_old = state_new
            if terminated or truncated:
                break
        agent.n_game += 1
        current_batch_loss = agent.train_long_memory()
        # Target network update
        if episode % config.target_update_interval == 0:
            agent.update_target_network()
            logger.info(f"Target network updated at episode {episode}")
        # Epsilon decay
        if episode % config.log_interval == 0:
            agent.epsilon = max(config.epsilon_end, agent.epsilon * config.epsilon_decay)
        # Learning Rate Scheduling Logic
        if episode > 0 and episode % config.lr_decay_episodes == 0:
            current_lr = agent.trainer.optimizer.param_groups[0]['lr']
            new_lr = current_lr * config.lr_decay_factor
            updated_lr = max(new_lr, config.min_learning_rate)
            for param_group in agent.trainer.optimizer.param_groups:
                param_group['lr'] = updated_lr
        # logger.info(f"Episode {episode}, Score: {cummulative_reward}, Epsilon: {agent.epsilon:.2f}")
        if wandb_run:
            wandb_run.log({
                "train/episode": episode,
                "train/score": cummulative_reward,
                "train/epsilon": agent.epsilon,
                "train/td_error": current_batch_loss
            })
        # Evaluation with epsilon=0
        if total_env_training_steps >= next_evaluation_at_step and wandb_run:
            eval_score = evaluate_policy(agent, eval_env, config, n_episodes=10)
            wandb_run.log({
                "eval/average_reward": eval_score,
                "eval/total_training_steps": total_env_training_steps,
            })
            next_evaluation_at_step += EVAL_FREQUENCY_STEPS 
        # Checkpointing
        if episode % config.checkpoint_interval == 0:
            logger.info(f"Saving checkpoint at episode {episode}")
            agent.save(f"ckpt_ep{episode}", run_id=run_id)
    agent.save('final_model', run_id=run_id)
    env.close()
    eval_env.close()
    logger.info("Training completed.")

def create_argument_parser():
    p = argparse.ArgumentParser(description="DQN Training Configuration")
    default = DQNConfig()
    p.add_argument("--total-episodes", type=int, default=default.total_episodes)
    p.add_argument("--batch-size", type=int, default=default.batch_size)
    p.add_argument("--memory-size", type=int, default=default.memory_size)
    p.add_argument("--learning-rate", type=float, default=default.learning_rate)
    p.add_argument("--gamma", type=float, default=default.gamma)
    p.add_argument("--epsilon-start", type=float, default=default.epsilon_start)
    p.add_argument("--epsilon-end", type=float, default=default.epsilon_end)
    p.add_argument("--epsilon-decay", type=float, default=default.epsilon_decay)
    p.add_argument("--hidden-dim", type=int, default=default.hidden_dim)
    p.add_argument("--checkpoint-dir", type=str, default=default.checkpoint_dir)
    p.add_argument("--log-interval", type=int, default=default.log_interval)
    p.add_argument("--seed", type=int, default=default.seed)
    p.add_argument("--map-name", type=str, default=default.map_name)
    return p


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    config = DQNConfig(**vars(args))
    wandb_run = wandb.init(
        project="medium-delivery-dqn",
        name=f"DQN-{config.map_name}-{config.total_episodes}ep",
        config=config.__dict__,
        tags=["dqn", "gymnasium"],
    )
    train(config, wandb_run)
    wandb_run.finish()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
