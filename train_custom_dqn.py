import torch
import random
import numpy as np
from collections import deque
from model_custom_dqn import Linear_QNetGen3, QTrainer
from environment2 import SimpleDeliveryEnv as Environment
from pathlib import Path

np.set_printoptions(linewidth=np.inf)
max_mem = 10_000
batch_size = 1000
lr = 0.001

class Agent:
    def __init__(self, observation_space, action_space):
        self.n_game = 0
        self.epsilon = 1.0
        self.gamma = 0.9  # <1
        self.memory = deque(maxlen=max_mem)
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_size = action_space.n
        self.observation_size = observation_space.shape[0]

        self.model = Linear_QNetGen3(self.observation_size, 128, self.action_size)
        # self.model.load_state_dict(torch.load('models/Gen3/model218.pth'))
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > batch_size:
            mini_sample = random.sample(self.memory, batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, done = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            final_move = random.randint(0, 1)
            # final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            final_move = torch.argmax(prediction).item()
            # final_move[move] = 1
        return final_move


def train():
    
    # env = Environment(
    #     # grid_fp=Path("grid_configs/A1_grid.npy"),
    #     render_mode='human',
    #     sigma=0.1,
    #     max_episode_steps=1000
    # )
    
    # # No gui mode
    env = Environment(
        render_mode='human',
    ) 
    
    agent = Agent(observation_space=env.observation_space, action_space=env.action_space) 

    # Get begin state
    state_old, info = env.reset()
    # env.render()

    old_reward = 0
    cummulative_reward = 0
    for i in range(1_000_000):
        if i % 100 == 0:
            # print(i)
            agent.epsilon = max(0.1, agent.epsilon * 0.99)

        # get move
        final_move = agent.get_action(state_old)

        # perform move
        state_new, reward, terminated, truncated, info = env.step(final_move)
        cummulative_reward += reward

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, terminated)

        # remember
        agent.remember(state_old, final_move, reward, state_new, terminated)

        state_old = state_new

        if terminated or truncated:  # Game terminated or truncated (e.g., max steps reached)
            # train long memory
            # score = env.world_stats["cumulative_reward"]
            score = cummulative_reward
            cummulative_reward = 0
            if terminated:
                print(f"Game ended with score: {score}")
            else:
                print("Game truncated, max steps reached.")
            state_old, info = env.reset()
            agent.n_game += 1
            agent.train_long_memory()

            print(f'Game {agent.n_game}, Score: {score}, Epsilon: {agent.epsilon:.2f}')
            if score > old_reward:
                agent.model.save(file_name='model{}.pth'.format(score))
                old_reward = score
    env.close()
    
if __name__ == '__main__':
    train()
