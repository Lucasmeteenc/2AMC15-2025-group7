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
    parser.add_argument('--n_episodes', type=int, default=4000,
                        help='Total number of training episodes')
    parser.add_argument('--reward_threshold', type=float, default=900,
                        help='Reward threshold to consider training successful')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum number of steps in the environment')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batchsize for training')

    args = parser.parse_args()
    return args


# TEST Environment ----------------------------- 
class EasyDeliveryBotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.fig, self.ax = None, None

        low = np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        self.action_space = spaces.Discrete(4)

        self.orientation = 0.0
        self.turn_angle = np.pi / 8
        self.step_size = 0.05
        self.position = np.array([0.1, 0.1], dtype=np.float32)
        self.goal = np.array([0.9, 0.9], dtype=np.float32)
        
        self.obstacles = [
            ((0.4, 0.4), (0.6, 0.6)),
        ]
        self.trail = []
        
        self.max_episode_steps = 10000
        self.current_step = 0

    def _is_collision(self, pos):
        for (low, high) in self.obstacles:
            if low[0] <= pos[0] <= high[0] and low[1] <= pos[1] <= high[1]:
                return True
        return False

    def _get_obs(self):
        return np.concatenate([
            self.position,
            np.array([np.cos(self.orientation), np.sin(self.orientation)]),
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([0.1, 0.1], dtype=np.float32)
        self.orientation = 0.0
        self.trail = [self.position.copy()]
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        new_position = self.position.copy()

        if action == 0: 
            delta = self.step_size * np.array([np.cos(self.orientation), np.sin(self.orientation)])
            new_position += delta
        elif action == 1: 
            self.orientation -= self.turn_angle
        elif action == 2:  
            self.orientation += self.turn_angle

        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi
        
        new_position = np.clip(new_position, 0.0, 1.0)

        if not self._is_collision(new_position):
            self.position = new_position
            if action == 0:
                self.trail.append(self.position.copy())

        #self.position = new_position
        if action == 0:
            self.trail.append(self.position.copy())

        distance_to_goal = np.linalg.norm(self.position - self.goal)
        done = False
        reward = -0.1 

        if distance_to_goal < 0.1:
            reward = 200.0 
            done = True

        truncated = self.current_step >= self.max_episode_steps

        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.fig.show()

        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect("equal")
        self.ax.set_title("Easy DeliveryBot Environment (Sanity Check)")

        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("#eaf7f5")

        self.ax.plot(*self.goal, 'g*', markersize=20, label="Goal")
        self.ax.plot(*self.position, 'bo', markersize=10, label="Robot")

        dx = 0.05 * np.cos(self.orientation)
        dy = 0.05 * np.sin(self.orientation)
        self.ax.arrow(self.position[0], self.position[1], dx, dy,
                      head_width=0.03, head_length=0.02, fc='b', ec='b')
        
        for (low, high) in self.obstacles:
            width = high[0] - low[0]
            height = high[1] - low[1]
            rect = patches.Rectangle(low, width, height, color="#686461")
            self.ax.add_patch(rect)

        if len(self.trail) > 1:
            trail_array = np.array(self.trail)
            points = trail_array.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='viridis', zorder=5)
            lc.set_array(np.arange(len(self.trail)))
            lc.set_linewidth(3)
            self.ax.add_collection(lc)
        
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1 / self.metadata["render_fps"])

    def close(self):
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            self.fig, self.ax = None, None

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
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
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

    states = torch.cat(states)
    actions = torch.cat(actions)
    actions_log_probability = torch.cat(actions_log_probability)
    values = torch.cat(values).squeeze(-1) 
    
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    
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
        for batch_idx, (states_b, actions_b, log_prob_old_b, advantages_b, returns_b) in enumerate(dataloader):

            action_pred_b, value_pred_b = agent(states_b)
            value_pred_b = value_pred_b.squeeze(-1)
            
            action_prob_b = f.softmax(action_pred_b, dim=-1)

            action_prob_b = action_prob_b + 1e-5 
            probability_distribution_new_b = distributions.Categorical(action_prob_b)
            entropy_b = probability_distribution_new_b.entropy()
            
            actions_log_probability_new_b = probability_distribution_new_b.log_prob(actions_b)
            
            surrogate_loss = calculate_surrogate_loss(log_prob_old_b,actions_log_probability_new_b,epsilon,advantages_b)
            policy_loss, value_loss = calculate_losses(surrogate_loss,entropy_b,entropy_coefficient,returns_b,value_pred_b)
            
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
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
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

def plot_train_rewards(train_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(train_rewards, label='Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(train_rewards), color='y', linestyle='--', label=f'Reward Threshold ({reward_threshold})')
    plt.legend(loc='lower right')
    plt.grid()
    plt.title('PPO Rewards Over Episodes')
    plt.show()

def plot_losses(policy_losses, value_losses):
    plt.figure(figsize=(12, 8))
    plt.plot(value_losses, label='Value Loss')
    plt.plot(policy_losses, label='Policy Loss')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('PPO Policy and Value Loss Over Episodes')
    plt.show()


def train_PPO(agent, train_env, eval_env, optimizer, args):
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
            avg_reward = np.mean(train_rewards[-10:])
            print(f"Ep: {episode:4d} | Avg. Reward: {np.mean(train_rewards[-10:]):.2f} | "
                  f"policy Loss: {avg_policy_loss:.4f} | value Loss: {avg_value_loss:.4f}")
            
            with open(train_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode, avg_reward, avg_policy_loss, avg_value_loss])

        if episode % 1000 == 0:
            print(f"\nEvaluating at Ep {episode}")
            eval_reward = evaluate(eval_env, agent, args.max_steps)
            print(f"Evaluation Reward: {eval_reward:.2f}\n")

            with open(eval_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode, eval_reward])

    return train_rewards, policy_losses, value_losses

if __name__ == "__main__":
    args = parse_args()

    train_env = EasyDeliveryBotEnv(render_mode=None) 

    eval_env = EasyDeliveryBotEnv(render_mode="human") 

    state_space_size = train_env.observation_space.shape[0] 
    action_space_size = train_env.action_space.n 

    agent = ActorCritic(state_space_size,args.hidden_dimensions,action_space_size,args.dropout)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    train_rewards, policy_losses, value_losses = [], [], []

    train_rewards, policy_losses, value_losses = train_PPO(agent, train_env, eval_env, optimizer, args)

    plot_train_rewards(train_rewards, args.reward_threshold)
    plot_losses(policy_losses, value_losses)

    print("\n--- Testing Evaluation with rendering ---")
    final_eval_reward = evaluate(eval_env, agent, args.max_steps)
    print(f"Final Evaluation Reward: {final_eval_reward:.2f}")

    obs, _ = eval_env.reset()
    done = False
 
    for _ in range(args.max_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = agent(obs_tensor)
            action_prob = f.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1).item()
        obs, _, done, _, _ = eval_env.step(action)
        eval_env.render()
        if done:
            break
    eval_env.close()
    train_env.close()