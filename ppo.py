import argparse
import os
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

def make_env(gym_id, idx, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = RecordEpisodeStatistics(env)
        if idx == 0:
            env = RecordVideo(env, f'videos/{run_name}', episode_trigger=lambda x: x % 1000 == 0)
        return env
    return thunk

def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Training Configuration')

    # General Settings
    parser.add_argument('--exp-name', type=str, default='ppo_experiment', help='Name of the experiment')
    parser.add_argument('--gym-id', type=str, default='CartPole-v1', help='Gym environment ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use-gpu', action=argparse.BooleanOptionalAction, default=True, help='Use GPU (cuda or mps) for training if available (e.g., --use-gpu or --no-use-gpu)')
    parser.add_argument('--use-wandb', action=argparse.BooleanOptionalAction, default=True, help='Use Weights & Biases for logging (e.g., --use-wandb or --no-use-wandb)')
    parser.add_argument('--wandb-project', type=str, default='ppo_experiment', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Weights & Biases entity (team) name (username by default)')

    # Environment Settings
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments to use')

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


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_name = f'{args.exp_name}_{args.gym_id}_seed{args.seed}__{int(time.time())}'
    print(f"Run Name: {run_name}")

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
        [make_env(args.gym_id, i, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "This script is designed for discrete action spaces only."

    agent = PPOAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    log_probs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    num_updates = args.total_timesteps // args.batch_size
    
    for update in range(1, num_updates + 1):
        # Anneal the learning rate linearly
        frac = 1.0 - (update - 1.0) / num_updates
        lr = args.learning_rate * frac
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Perform policy rollouts
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action.cpu()
            log_probs[step] = log_prob.cpu()

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(
                np.logical_or(terminated, truncated),
                dtype=torch.float32, device=device
            )

            # Log episode data if available
            if "final_info" in info:
                for final_info_item in info["final_info"]:
                    # Skip items that are None (environments that didn't just end)
                    if final_info_item is None:
                        continue

                    # Check for episode data in the final_info_item
                    if "episode" in final_info_item:
                        episode_return = final_info_item["episode"]["r"]
                        episode_length = final_info_item["episode"]["l"]
                        
                        print(f"global_step={global_step}, episode_return={episode_return}, episode_length={episode_length}")
                        writer.add_scalar("charts/episode_return", episode_return, global_step)
                        writer.add_scalar("charts/episode_length", episode_length, global_step)

        # Compute advantages and returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten() # Bootstrap if not done
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelambda = 0
            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[step + 1]
                    next_values = values[step + 1]

                delta = rewards[step] + args.gamma * next_values * next_non_terminal - values[step]
                advantages[step] = lastgaelambda = delta + args.gamma * args.gae_lambda * next_non_terminal * lastgaelambda
            returns = advantages + values

        # Flatten the batch
        obs_batch = obs.reshape((-1, ) + envs.single_observation_space.shape)
        log_probs_batch = log_probs.reshape(-1)
        actions_batch = actions.reshape((-1,) + envs.single_action_space.shape)
        advantages_batch = advantages.reshape(-1)
        returns_batch = returns.reshape(-1)
        values_batch = values.reshape(-1)

        # Optimize the policy and value function
        indices_batch = np.arange(args.batch_size)
        clipped_fracs = []
        for epoch in range(args.num_epochs):
            np.random.shuffle(indices_batch)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                indices_mb = indices_batch[start:end]

                # forward pass on minibatch observations
                _, log_probs_mb, entropy_mb, values_mb = agent.get_action_and_value(
                    obs_batch[indices_mb], actions_batch[indices_mb]
                )
                logratio = log_probs_mb - log_probs_batch[indices_mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approximate KL divergence for debugging
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipped_fracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # advantage normalization
                advantages_mb = advantages_batch[indices_mb]
                advantages_mb = (advantages_mb - advantages_mb.mean()) / (advantages_mb.std() + 1e-8)
                
                # policy loss (+ clipping)
                policy_loss = advantages_mb * ratio
                policy_loss_clipped = advantages_mb * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                policy_loss = torch.min(policy_loss, policy_loss_clipped).mean()

                # value loss (+ clipping)
                v_loss_unclipped = (values_mb - returns_batch[indices_mb]).pow(2)
                v_clipped = values_batch[indices_mb] + torch.clamp(
                    values_mb - values_batch[indices_mb],
                    -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - returns_batch[indices_mb]).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # entropy loss
                entropy_loss = entropy_mb.mean()

                # overall loss
                loss = policy_loss - args.entropy_coef * entropy_loss + args.value_loss_coef * value_loss

                # optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Calculate explained variance
        with torch.no_grad():
            y_pred, y_true = values_batch.cpu().numpy(), returns_batch.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log training metrics
        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('charts/value_loss', value_loss.item(), global_step)
        writer.add_scalar('charts/policy_loss', policy_loss.item(), global_step)
        writer.add_scalar('charts/entropy_loss', entropy_loss.item(), global_step)
        writer.add_scalar('charts/approx_kl', approx_kl, global_step)
        writer.add_scalar('charts/clipped_fraction', np.mean(clipped_fracs), global_step)
        writer.add_scalar('charts/explained_variance', explained_var, global_step)
        print(f'SPS: {global_step / (time.time() - start_time):.2f} steps/sec')
        writer.add_scalar('charts/sps', global_step / (time.time() - start_time), global_step)

    envs.close()
    writer.close()

    if args.use_wandb:
        run.finish()