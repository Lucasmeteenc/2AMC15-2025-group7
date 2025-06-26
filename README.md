In this assignment we are asked to implement a variety of reinforcement learning algorithms that are going to assist us in identifying the optimal policy a delivery robot should follow.

The algorithms that were implemented are:
1. DQN
2. PPO


# Spatial Heatmap Visualization

## Overview
Two nearly identical scripts for visualizing agent trajectories as continuous spatial heat-maps in the `MediumDeliveryEnv`.  
- **dqn_spatial_heatmap.py**: uses a pre‐trained DQN agent.  
- **ppo_spatial_heatmap.py**: uses a pre‐trained PPO agent.
Both scripts share the same interface and differ only in the RL algorithm and default KDE settings.

## Requirements
- matplotlib  
- seaborn  
- (plus any dependencies of `dqn_agent`, `ppo_agent`, and `environments/medium_delivery_env`)

## Usage
```bash
python dqn_spatial_heatmap.py [--model PATH] [--map NAME] [--trials N]
                             [--bw FLOAT] [--cmap NAME] [--out FILE] [-v]

python ppo_spatial_heatmap.py [--model PATH] [--map NAME] [--trials N]
                             [--bw FLOAT] [--cmap NAME] [--out FILE] [-v]
```

## Arguments
--model: path to your .pth file
--map: one of the keys in MAIL_DELIVERY_MAPS
--trials: number of episodes (default: 100)
--bw: KDE bandwidth adjust (DQN default 0.3, PPO default 0.5)
--cmap: matplotlib colormap (e.g. viridis, magma)
--out: output PNG filename
-v: verbose logging

## Output
Heat-map PNGs are saved under sheatmap_results/<out>.


### AI Disclaimer

AI tools were used only to assist with:
1. Debugging code-related issues
2. Generating code snippets for graph visualization and plotting
