import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNetGen3, QTrainer
from world.environment import Environment
from pathlib import Path

np.set_printoptions(linewidth=np.inf)
max_mem = 10_000
batch_size = 1000
lr = 0.001

def train():
    # GUI variant:
    # env = Environment(
    #     grid_fp=Path("grid_configs/A1_grid.npy"),
    #     render_mode='human',
    #     sigma=0.1,
    #     max_episode_steps=1000,
    #     random_seed=1
    # )
    
    # # No gui mode
    env = Environment(
        grid_fp=Path("grid_configs/A1_grid.npy"),
        render_mode='rgb_array',
        sigma=0.1,
        max_episode_steps=1000,
        random_seed=1
    ) 
    
    # Get begin state
    state_old, info = env.reset()
    # env.render()

    old_reward = 0
    for i in range(1_000_000):
        # Always try to charge
        final_move = 4  # TODO agent?

        # perform move
        state_new, reward, terminated, truncated, info = env.step(final_move)

        state_old = state_new

        if terminated or truncated:  # Game terminated or truncated (e.g., max steps reached)
            # train long memory
            score = env.world_stats["cumulative_reward"]
            state_old, info = env.reset()

            print(f'Epsiode Score: {score}')
            if score > old_reward:
                old_reward = score
    env.close()
    
if __name__ == '__main__':
    train()
