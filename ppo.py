import argparse
import random
import numpy as np
import time
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.categorical as Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.record_video import RecordVideo

from world.environment import Environment
from pathlib import Path

def make_custom_env(idx, run_name, grid_fp, max_episode_steps=1000, render_mode='rgb_array'):
    def thunk():
        env = Environment(grid_fp=grid_fp, max_episode_steps=max_episode_steps, render_mode=render_mode)
        env = RecordEpisodeStatistics(env)
        if idx == 0:
            env = RecordVideo(env, f'videos/{run_name}', episode_trigger=lambda x: x % 100 == 0)
        return env
    return thunk

def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Training Configuration')

    # General Settings
    parser.add_argument('--exp-name', type=str, default='ppo_custom_env', help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use-gpu', action=argparse.BooleanOptionalAction, default=True, help='Use GPU (cuda or mps) for training if available (e.g., --use-gpu or --no-use-gpu)')
    parser.add_argument('--use-wandb', action=argparse.BooleanOptionalAction, default=True, help='Use Weights & Biases for logging (e.g., --use-wandb or --no-use-wandb)')
    parser.add_argument('--wandb-project', type=str, default='ppo_experiment', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Weights & Biases entity (team) name (username by default)')

    # Environment Settings
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments to use')
    parser.add_argument('--grid-fp', type=str, default='grid_configs/A1_grid.npy', help='Grid file path for the environment')
    parser.add_argument('--max-episode-steps', type=int, default=1000, help='Maximum number of steps per episode') # might be duplicate with the num-steps in the training parameters

    # Training Parameters
    parser.add_argument('--total-timesteps', type=int, default=25000, help='Total number of timesteps for training')
    parser.add_argument('--num-steps', type=int, default=128, help='Number of steps to run in each environment per update')
    parser.add_argument('--num-minibatches', type=int, default=4, help='Number of minibatches to split the batch into for training')
    parser.add_argument('--num-epochs', type=int, default=4, help='Number of epochs to train on each minibatch')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Lambda for Generalized Advantage Estimation')
    parser.add_argument('--clip-coef', type=float, default=0.2, help='Clipping coefficient for PPO')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Coefficient for entropy bonus')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='Coefficient for value loss')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm for clipping')

    args = parser.parse_args()

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches

    return args

class PPOAgent(nn.Module):
    def __init__(self, envs):
        super(PPOAgent, self).__init__()
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), gain=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), gain=0.01)
        )

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(x)
        return action, log_prob, entropy, value

class PPOTrainer:
    def __init__(self, args, envs, agent, optimizer, writer, device='cpu'):
        self.args = args
        self.envs = envs
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer
        self.device = device

        # --- Rollout Buffer Initialization ---
        self.obs = torch.zeros((args.num_steps, args.num_envs) + self.envs.single_observation_space.shape, device=self.device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + self.envs.single_action_space.shape, device=self.device)
        self.log_probs = torch.zeros((args.num_steps, args.num_envs), device=self.device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs), device=self.device)
        self.dones = torch.zeros((args.num_steps, args.num_envs), device=self.device)
        self.values = torch.zeros((args.num_steps, args.num_envs), device=self.device)

    def train(self):
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.args.num_envs, device=self.device)
        num_updates = self.args.total_timesteps // self.args.batch_size
        
        for update in range(1, num_updates + 1):
            # Anneal the learning rate linearly
            frac = 1.0 - (update - 1.0) / num_updates
            self.optimizer.param_groups[0]['lr'] = self.args.learning_rate * frac
            
            # Perform policy rollouts
            for step in range(0, self.args.num_steps):
                global_step += 1 * self.args.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.log_probs[step] = log_prob

                next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                self.rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device)
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                next_done = torch.tensor(
                    np.logical_or(terminated, truncated), 
                    dtype=torch.float32, 
                    device=self.device
                )

                # Log episode data if available
                if '_episode' in info:
                    finished_mask = info['_episode']

                    episode_returns = info['episode']['r'][finished_mask]
                    episode_lengths = info['episode']['l'][finished_mask]

                    for i in range(len(episode_returns)):
                        print(f"global_step={global_step}, episode_return={episode_returns[i]}, episode_length={episode_lengths[i]}")
                        self.writer.add_scalar("charts/episode_return", episode_returns[i], global_step)
                        self.writer.add_scalar("charts/episode_length", episode_lengths[i], global_step)

            # Compute advantages and returns
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).flatten() # Bootstrap if not done
                advantages = torch.zeros_like(self.rewards, device=self.device)
                lastgaelambda = 0
                for t in reversed(range(self.args.num_steps)):
                    next_non_terminal = 1.0 - (self.dones[t+1] if t < self.args.num_steps - 1 else next_done)
                    next_values = self.values[t+1] if t < self.args.num_steps - 1 else next_value
                    delta = self.rewards[t] + self.args.gamma * next_values * next_non_terminal - self.values[t]
                    advantages[t] = lastgaelambda = delta + self.args.gamma * self.args.gae_lambda * next_non_terminal * lastgaelambda
                returns = advantages + self.values

            # Flatten the batch
            obs_b = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            log_probs_b = self.log_probs.reshape(-1)
            actions_b = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            advantages_b = advantages.reshape(-1)
            returns_b = returns.reshape(-1)
            values_b = self.values.reshape(-1)

            # Optimize the policy and value function
            indices = np.arange(self.args.batch_size)
            clipped_fracs = []
            for epoch in range(self.args.num_epochs):
                np.random.shuffle(indices)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    indices_mb = indices[start:end]

                    # forward pass on minibatch observations
                    _, new_log_prob, entropy_mb, values_mb = self.agent.get_action_and_value(obs_b[indices_mb], actions_b[indices_mb])
                    logratio = new_log_prob - log_probs_b[indices_mb]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approximate KL divergence for debugging
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        clipped_fracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                        self.writer.add_scalar('charts/approx_kl', approx_kl, global_step)
                        self.writer.add_scalar('charts/clipped_fraction', np.mean(clipped_fracs), global_step)

                    # advantage normalization
                    advantages_mb = advantages_b[indices_mb]
                    advantages_mb = (advantages_mb - advantages_mb.mean()) / (advantages_mb.std() + 1e-8)
                    
                    # policy loss (+ clipping)
                    policy_loss = advantages_mb * ratio
                    policy_loss_clipped = advantages_mb * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    policy_loss = torch.min(policy_loss, policy_loss_clipped).mean()

                    # value loss (+ clipping)
                    v_loss_unclipped = (values_mb - returns_b[indices_mb]).pow(2)
                    v_clipped = values_b[indices_mb] + torch.clamp(
                        values_mb - values_b[indices_mb],
                        -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - returns_b[indices_mb]).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # entropy loss
                    entropy_loss = entropy_mb.mean()

                    # overall loss
                    loss = - policy_loss - args.entropy_coef * entropy_loss + args.value_loss_coef * value_loss

                    # optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            # Calculate explained variance
            y_pred, y_true = values_b.cpu().numpy(), returns_b.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Log training metrics    
            self.writer.add_scalar('charts/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
            self.writer.add_scalar('charts/value_loss', value_loss.item(), global_step)
            self.writer.add_scalar('charts/policy_loss', policy_loss.item(), global_step)
            self.writer.add_scalar('charts/entropy_loss', entropy_loss.item(), global_step)
            self.writer.add_scalar('charts/approx_kl', approx_kl, global_step)
            self.writer.add_scalar('charts/clipped_fraction', np.mean(clipped_fracs), global_step)
            self.writer.add_scalar('charts/explained_variance', explained_var, global_step)
            print(f'SPS: {int(global_step / (time.time() - start_time))} steps/sec')
            self.writer.add_scalar('charts/sps', int(global_step / (time.time() - start_time)), global_step)

        self.close()

    def close(self):
        self.envs.close()
        self.writer.close()

if __name__ == "__main__":
    args = parse_args()
    run_name = f'{args.exp_name}_custom-env_seed{args.seed}__{int(time.time())}'

    # Convert grid file path to Path object
    args.grid_fp = Path(args.grid_fp)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine device for training
    if torch.cuda.is_available() and args.use_gpu:
        device = "cuda"
    elif torch.backends.mps.is_available() and args.use_gpu:
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Initialize Weights & Biases if enabled
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,  # Automatically sync TensorBoard logs
            monitor_gym=True,       # Automatically log videos of the environment
            save_code=True          # Save the main script to W&B
        )
        print(f"W&B run initialized: {wandb.run.name}")

    # Logging setup using TensorBoard (run `tensorboard --logdir=runs` to view)
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text(
        'experiment_info',
        '|Parameter|Value|\n|---|---|\n' +
        '\n'.join([f'|{k}|{v}|' for k, v in vars(args).items()])
    )

    # Training logic    
    envs = gym.vector.SyncVectorEnv(
        [make_custom_env(i, run_name, args.grid_fp, args.max_episode_steps) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "This script is designed for discrete action spaces only."

    agent = PPOAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # trainer = PPOTrainer(agent, envs, optimizer, args, writer, device)
    trainer = PPOTrainer(args, envs, agent, optimizer, writer, device)
    trainer.train()

    if args.use_wandb:
        run.finish()