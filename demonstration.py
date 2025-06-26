# %%
#########################
# Demonstration of the project
# 
# This script demonstrates how you can run the different files and derive the results that we created in the report
# This script consists of stand-alone cells that can be run individually or in sequence
# By setting the seeds correctly you can observe that the results are fully reproducible.
###
from environments.medium_delivery_env import MediumDeliveryEnv as Environment # The gymnasium enviornment
import maps # The maps
MAPS = maps.MAIL_DELIVERY_MAPS

import torch
from IPython.display import Video  # For displaying videos in interactive window (e.g. Jupyter Notebook)

import dqn_agent
import dqn_test

import ppo_agent
import ppo_test  

import sys
sys.modules['__main__'].PPOConfig = ppo_agent.PPOConfig

BEST_MODELS_DIR = 'best_models'  # Directory where the best models are saved
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
FIXED_SEED = 42 
map_name = 'default'  # Default map for the demonstration 
ppo_agent.set_random_seeds(FIXED_SEED)  # Set the random seed for reproducibility
dqn_agent.set_random_seeds(FIXED_SEED)  # Set the random seed for reproducibility

# %% 
#########################
# Enviornment showcase
#
# As described in our report, the enviornment is defined by obstacles, a pick-up point and a drop-off point.
# The robot itself can move forward, turn left, or turn right
# The robot observes it's orientation and rays, which tell how close it is to colliding with a wall
#
# Below, you can see a run of our best PPO agent for the default map
# And a run of DQN on the inside map (which performs less well), 
# demonstrating the different enviornments
###
def demo_best_agent_on_map(algorithm, map_name, video_dir='demo_videos', device='cpu'):
    print(f"Running demo for {algorithm.upper()} on map '{map_name}'")
    env = Environment(
        render_mode='rgb_array', 
        map_config=MAPS[map_name],
        seed=FIXED_SEED
    )
    
    # Best checkpoints are located in: best_models/map_name/algorithm.pth
    checkpoint_path = f'{BEST_MODELS_DIR}/{map_name}/{algorithm}.pth'
    
    # Load and test the model
    if algorithm == 'ppo':
        actor = ppo_test.load_model(checkpoint_path, env.observation_space.shape[0], env.action_space.n, device) 
        ppo_agent.set_random_seeds(FIXED_SEED)  # Set the random seed for reproducibility
        total_reward, video_path = ppo_test.test_model(actor, env, device, video_dir=f'{video_dir}/fully_trained_map_{map_name}')
    else:
        config = dqn_test.DQNConfig(map_name=map_name)
        actor = dqn_test.load_model(checkpoint_path, config, device)
        dqn_agent.set_random_seeds(FIXED_SEED)  # Set the random seed for reproducibility
        total_reward, video_path = dqn_test.test_model(actor, env, device, video_dir=f'{video_dir}/fully_trained_map_{map_name}')

    # Test the model
    print(f"Total reward: {total_reward}")
    print(f"Video saved at: {video_path}")
    return video_path, actor


# %%
# Map 1 default
map_name = 'default'  # Choose the map you want to evaluate on, # can be 'default', 'inside'
algorithm = 'ppo'  # Choose the algorithm you want to evaluate on, # can be 'dqn' or 'ppo'

# Test model
video_path, best_actor = demo_best_agent_on_map(algorithm, map_name, video_dir='demo_videos', device=DEVICE)

# Display the video
print("Trying to display the video... (won't display when running in a script)")
Video(video_path, embed=True, width=800, height=600) if video_path else None

# %% 
# Map 2 inside 
# DQN Agent
map_name = 'inside' # Options: 'default' or 'inside'
algorithm = 'ppo'  # Options: 'dqn' or 'ppo'

# Test model
video_path, best_actor = demo_best_agent_on_map(algorithm, map_name, video_dir='demo_videos', device=DEVICE)

# Display the video
Video(video_path, embed=True, width=800, height=600) if video_path else None


# %%
#########################
# Agent Training
#
# This part of the code shows how we train our agent.
# We chose to implement both DQN and PPO.
# We will demonstrate how we trained and evaluated both
# We will restrict the demo to the default map, but you can it using 
###
map_name = 'default'  # Choose the map you want to evaluate on, can be 'default', 'inside', or 'empty' (empty is easier)

# %%
#########################
# DQN Agent Training
#
# This part of the code shows how we train our agent.
# We use the optimal hyper parameters that we found in the report.
# We only train for 1000 episodes to keep the demo short
###

# Create agent from scratch
checkpoint_dir = f'demo_videos/checkpoints_dqn/{map_name}'
config = dqn_agent.DQNConfig(map_name=map_name, 
                             seed=FIXED_SEED, 
                             total_episodes=1000,  # Reduce the number of episodes for faster training in demo
                             checkpoint_dir=checkpoint_dir)

# Train the agent
print(f"Training DQN agent on map '{map_name}' with seed {FIXED_SEED}...")
print("This may take a while...")
dqn_agent.train(config)
print(f"Training complete. Results saved to {checkpoint_dir}")


# %%
#########################
# DQN Agent Demo run
#
# This demo shows our shortly trained agent running the game
###
# Load the trained model from the code above
print(f"Loading DQN model from {f'demo_videos/checkpoints_dqn/{map_name}/model_final_model.pth'}")
actor = dqn_test.load_model(
    model_path=f'demo_videos/checkpoints_dqn/{map_name}/model_final_model.pth',
    config=dqn_agent.DQNConfig(map_name=map_name, seed=FIXED_SEED),
    device=DEVICE
)

dqn_agent.set_random_seeds(FIXED_SEED)  # Set the random seed for reproducibility
env = dqn_agent.create_environment(dqn_agent.DQNConfig(map_name=map_name, seed=FIXED_SEED), seed=FIXED_SEED, render_mode='rgb_array')
video_dir = f'demo_videos/shortly_trained_map_{map_name}'  # Directory to save the video


# Test the model
print(f"Testing DQN agent on map '{map_name}'...")
total_reward, video_path = dqn_test.test_model(
    actor,
    env,
    device=DEVICE,
    video_dir=video_dir,
    video_name=f'dqn_demo_{map_name}'
)

# Display the video
print(f"Total reward: {total_reward}")
print(f"Video saved at: {video_path}")
Video(video_path, embed=True, width=800, height=600) if video_path else None

# As can be seen, after 1000 episodes, the agent prevents collisions but misses the pick-up and drop-off points.

# %%
#########################
# PPO Agent Training
#
# This part of the code shows how we train our agent.
# We use the optimal hyper parameters that we found in the report.
###

# Init agent with the best hyper parameters.
# Switch to CPU to ensure determinism in the demo
DEVICE = torch.device("cpu")  # Use CPU for this demo to ensure determinism
config = ppo_agent.PPOConfig(map_name=map_name,
                             checkpoint_dir=f'demo_videos/checkpoints_ppo/{map_name}',
                             seed=FIXED_SEED,
                             total_timesteps=100000,  # Reduce the number of timesteps for faster training in demo
                             num_envs=1  # Use multiple environments for vectorized training
                             ) 
ppo_agent.set_random_seeds(FIXED_SEED)  # Set the random seed for reproducibility

# 1. Vectorised training environments
train_envs = ppo_agent.make_vec_env(config, config.num_envs, config.seed)

# 2. Dimensionality from ONE env instance
state_space_size = train_envs.single_observation_space.shape[0]
action_space_size = train_envs.single_action_space.n

# 3. PPO Agent
agent = ppo_agent.PPOAgent(config, DEVICE)
agent.setup_networks(state_space_size, action_space_size)

# 4. Training loop
print(f"Training PPO agent on map '{map_name}'")
train_returns = agent.train(train_envs, save_update=False)  # Train update, but do not save intermediate models in the demo
train_envs.close()
print(f"Training complete. Results saved to {config.checkpoint_dir}")


# %%
#########################
# PPO Agent Demo run
#
# This demo shows our shortly trained agent running the game
###
# Load the trained model from the code above
print(f"Loading PPO model from {f'demo_videos/checkpoints_ppo/{map_name}/final_model_{map_name}.pth'}")
actor = ppo_test.load_model(
    model_path=f'demo_videos/checkpoints_ppo/{map_name}/final_model_no_wandb.pth',
    state_size=state_space_size,
    action_size=action_space_size,
    device=DEVICE
)

ppo_agent.set_random_seeds(FIXED_SEED)  # Set the random seed for reproducibility
env = Environment(
    render_mode='rgb_array', 
    map_config=MAPS[map_name],
    seed=FIXED_SEED
)

print(f"Testing PPO agent on map '{map_name}'")
video_dir = f'demo_videos/shortly_trained_map_{map_name}'  # Directory to save the video
# Test the model
total_reward, video_path = ppo_test.test_model(
    actor,
    env,
    device=DEVICE,
    video_dir=video_dir
)

# Display the video
print(f"Total reward: {total_reward}")
print(f"Video saved at: {video_path}")
Video(video_path, embed=True, width=800, height=600) if video_path else None
# In this demo, the agent picks up the parcel but fails to deliver it. 
# After more training it should be able to do so


# %%
