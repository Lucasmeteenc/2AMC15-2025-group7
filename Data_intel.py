import torch
import random
import numpy as np
from learning_game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, Linear_QNetGen3, QTrainer
import gymnasium as gym


np.set_printoptions(linewidth=np.inf)
max_mem = 10_000
batch_size = 1000
lr = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 1.0
        self.gamma = 0.9  # <1
        self.memory = deque(maxlen=max_mem)

        self.model = Linear_QNetGen3(4, 128, 2)
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
    agent = Agent()
    env = gym.make("CartPole-v1", render_mode="human")

    # Get begin state
    state_old, info = env.reset()
    # env.render()

    score = 0
    old_reward = 0
    for i in range(10_000):
        if i % 100 == 0:
            # print(i)
            agent.epsilon = max(0.1, agent.epsilon * 0.95)

        # get move
        final_move = agent.get_action(state_old)

        # perform move
        state_new, reward, terminated, truncated, info = env.step(final_move)
        score += 1

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, terminated)

        # remember
        agent.remember(state_old, final_move, reward, state_new, terminated)

        state_old = state_new

        if terminated:
            # train long memory
            state_old, info = env.reset()
            agent.n_game += 1
            agent.train_long_memory()

            print(score)
            if score > old_reward:
                agent.model.save(file_name='model{}.pth'.format(score))
                old_reward = score

            score = 0
    env.close()
if __name__ == '__main__':
    train()
