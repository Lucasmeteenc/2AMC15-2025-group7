import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as distributions
import argparse
import os, csv

# from environment2 import SimpleDeliveryEnv
from environments.simple_delivery_env import SimpleDeliveryEnv

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Training Configuration')

    parser.add_argument('--hidden_dimensions', type=int, default=256,
                        help='Number of hidden units in the network')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for regularization')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='Learning rate for optimizer')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='Discount factor (gamma) for future rewards')
    parser.add_argument('--ppo_steps', type=int, default=4,
                        help='Number of PPO epochs per update')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Clipping parameter epsilon for PPO')
    parser.add_argument('--entropy_coefficient', type=float, default=0.01,
                        help='Entropy coefficient to encourage exploration')
    parser.add_argument('--n_episodes', type=int, default=1000,
                        help='Total number of training episodes')
    parser.add_argument('--reward_threshold', type=float, default=5000,
                        help='Reward threshold to consider training successful')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Maximum number of steps in the environment')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batchsize for training')

    args = parser.parse_args()
    return args

#------------------- PPO ---------------------

class ActorCritic(nn.Module):
    """
    Creates an actor critic model that combines both actor and critic networks.
    """
    def __init__(self, in_features, hidden_dimensions, action_dim, dropout):
        super().__init__()
        # actor
        self.actor_fc1 = nn.Linear(in_features, hidden_dimensions)
        self.actor_fc2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.actor_out = nn.Linear(hidden_dimensions, action_dim)
        self.actor_dropout = nn.Dropout(dropout)
        
        # critic
        self.critic_fc1 = nn.Linear(in_features, hidden_dimensions)
        self.critic_fc2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.critic_out = nn.Linear(hidden_dimensions, 1)
        self.critic_dropout = nn.Dropout(dropout)

    def forward(self, state):
        # actor forward
        a = f.relu(self.actor_fc1(state))
        a = self.actor_dropout(a)
        a = f.relu(self.actor_fc2(a))
        a = self.actor_dropout(a)
        action_pred = self.actor_out(a)

        # critic forward
        c = f.relu(self.critic_fc1(state))
        c = self.critic_dropout(c)
        c = f.relu(self.critic_fc2(c))
        c = self.critic_dropout(c)
        value_pred = self.critic_out(c)
        return action_pred, value_pred

def calculate_returns(rewards, discount_factor):
    """
    Calculates the discounted returns for the rewards.
    """
    rewards = torch.tensor(rewards, dtype=torch.float32)
    returns = torch.zeros_like(rewards)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + discount_factor * cumulative
        returns[i] = cumulative
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    return returns

def calculate_advantages(returns, values):
    """
    Calculates adgantages using Generalized Advantage Estimation (GAE).
    The advantage is used later to weight the log probability ration between the old and new policies."""
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    return advantages

def calculate_surrogate_loss(actions_log_probability_old,actions_log_probability_new,epsilon,advantages):
    """
    Calculates the clipple loss
    """
    advantages = advantages.detach()
    ratio = (actions_log_probability_new - actions_log_probability_old).exp()
    unclipped_loss = ratio * advantages
    clipped_loss = torch.clamp(ratio, min=1.0 - epsilon, max=1.0 + epsilon) * advantages 
    return torch.min(unclipped_loss, clipped_loss)

def calculate_losses(surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
    """
    Combines policy and value losses with entropy bonus
    """
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss + entropy_bonus).sum()
    value_loss = f.smooth_l1_loss(value_pred, returns).sum() # huber
    return policy_loss, value_loss

def forward_pass(env, agent, discount_factor, max_steps= 10000):
    """
    Model inference and data collection for PPO.
    """
    states, actions, actions_log_probability, values, rewards, episode_reward = [], [], [], [], [], 0.0
    state, _ = env.reset() 
    
    agent.train()
    
    for _ in range(max_steps): 
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        action_logits, value_logits = agent(state_tensor)
        action_prob = f.softmax(action_logits, dim=-1)
        
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated 

        states.append(state_tensor)
        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_logits)
        rewards.append(reward)
        episode_reward += reward
        
        state = next_state 

        if done:
            break 

    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    actions_log_probability = torch.cat(actions_log_probability).to(device)
    values = torch.cat(values).squeeze(-1).to(device)
    
    returns = calculate_returns(rewards, discount_factor).to(device)
    advantages = calculate_advantages(returns, values).to(device)
    
    return episode_reward, states, actions, actions_log_probability, advantages, returns


def update_policy(agent,states,actions,log_prob_old,advantages,returns,optimizer,ppo_steps,epsilon,entropy_coefficient, batch_size):
    total_policy_loss = 0
    total_value_loss = 0
    
    log_prob_old = log_prob_old.detach()
    actions = actions.detach()
    advantages = advantages.detach()
    returns = returns.detach() 

    dataset = TensorDataset(states,actions,log_prob_old,advantages,returns)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True) 

    for _ in range(ppo_steps):
        for batch_idx, (states_batch, actions_batch, log_prob_old_batch, advantages_batch, returns_batch) in enumerate(dataloader):
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)
            log_prob_old_batch = log_prob_old_batch.to(device)
            advantages_batch = advantages_batch.to(device)
            returns_batch = returns_batch.to(device)

            action_pred_batch, value_pred_batch = agent(states_batch)
            value_pred_batch = value_pred_batch.squeeze(-1)
            
            action_prob_batch = f.softmax(action_pred_batch, dim=-1)

            action_prob_batch = action_prob_batch + 1e-5 
            probability_distribution_new_batch = distributions.Categorical(action_prob_batch)
            entropy_batch = probability_distribution_new_batch.entropy()
            
            actions_log_probability_new_b = probability_distribution_new_batch.log_prob(actions_batch)
            
            surrogate_loss = calculate_surrogate_loss(log_prob_old_batch,actions_log_probability_new_b,epsilon,advantages_batch)
            policy_loss, value_loss = calculate_losses(surrogate_loss,entropy_batch,entropy_coefficient,returns_batch,value_pred_batch)
            
            optimizer.zero_grad()

            policy_loss.backward(retain_graph=True) 
            value_loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
    num_batches = len(dataloader)
    avg_policy_loss = total_policy_loss / (ppo_steps * num_batches)
    avg_value_loss = total_value_loss / (ppo_steps * num_batches)

    return avg_policy_loss, avg_value_loss

def evaluate(env, agent, args_max_steps=10000):
    agent.eval()

    episode_reward = 0
    state, _ = env.reset()

    for _ in range(args_max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_pred, _ = agent(state_tensor)
            action_prob = f.softmax(action_pred, dim=-1)
        
        action = torch.argmax(action_prob, dim=-1)
        # dist = distributions.Categorical(action_prob)
        
        # action = dist.sample()
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        episode_reward += reward
        state = next_state
        
        if done:
            break
    return episode_reward

def train(agent, train_env, eval_env, optimizer, args):
    train_rewards = []
    policy_losses = []
    value_losses = []

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    train_log_file = os.path.join(log_dir, "training_log.csv")
    eval_log_file = os.path.join(log_dir, "evaluation_log.csv")

    with open(train_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "average_reward", "policy_loss", "value_loss"])

    with open(eval_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "evaluation_reward"])

    print(f"PPO training for {args.n_episodes} episodes")

    for episode in range(1, args.n_episodes + 1):
        episode_reward, states, actions, actions_log_probability_old, advantages, returns = forward_pass(train_env, agent, args.discount_factor, args.max_steps)

        avg_policy_loss, avg_value_loss = update_policy(agent,states,actions,actions_log_probability_old,advantages,returns,optimizer,args.ppo_steps,args.epsilon,args.entropy_coefficient,args.batch_size)

        train_rewards.append(episode_reward)
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)

        if episode % 200 == 0:
            avg_reward = np.mean(train_rewards[-50:])
            print(f"Ep: {episode:4d} - Avg. Reward: {np.mean(train_rewards[-50:]):.2f} - "
                  f"policy Loss: {avg_policy_loss:.4f} - value Loss: {avg_value_loss:.4f}")
            
            with open(train_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode, avg_reward, avg_policy_loss, avg_value_loss])

        # if episode % 1000 == 0:
        #     print(f"\nEvaluating at Ep {episode}")
        #     eval_reward = evaluate(eval_env, agent, args.max_steps)
        #     print(f"Evaluation Reward: {eval_reward:.2f}\n")

        #     with open(eval_log_file, "a", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow([episode, eval_reward])

    return train_rewards, policy_losses, value_losses

def make_env():
    return SimpleDeliveryEnv(render_mode=None)

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_env = SimpleDeliveryEnv(render_mode=None) 

    eval_env = SimpleDeliveryEnv(render_mode="human") 

    state_space_size = train_env.observation_space.shape[0] 
    action_space_size = train_env.action_space.n 

    agent = ActorCritic(state_space_size,args.hidden_dimensions,action_space_size,args.dropout)
    agent.to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    train_rewards, policy_losses, value_losses = [], [], []

    train_rewards, policy_losses, value_losses = train(agent, train_env, eval_env, optimizer, args)

    print("\n--- Testing Evaluation with rendering ---")
    final_eval_reward = evaluate(eval_env, agent, args.max_steps)
    print(f"Evaluation Reward: {final_eval_reward:.2f}")

    obs, _ = eval_env.reset()
    done = False

    eval_env.close()
    train_env.close()