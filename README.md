In this assignment we are asked to implement a variety of reinforcement learning algorithms that are going to assist us in identifying the optimal policy a delivery robot should follow.

The algorithms that were implemented are:
1. DQN
2. PPO

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```


# Demonstration

The `demonstration.py` script provides a step-by-step walkthrough of our environment, maps, and trained agents. It highlights both testing and training workflows for our PPO and DQN models.

## Features
`demonstration.py` has several section in the code enabling you to see the following:

1. **Environment Visualization + Agent Performance showcase**

    * Demonstrate enviornment, lidar ray interacion and robot's action.
   * See our optimal pre-trained PPO and DQN agents
   * Change agent and map (see the comments in the file on how to do so)

3. **Interactive Training Demos**

   * Execute training loops for PPO and DQN agents directly within the script.
   * Due to shortened training epochs (for demonstration purposes), you will see agents on their way to learning optimal paths, though not yet fully converged.

4. **Video Recording**

   * Automatically saves demo videos to the `/demo_videos/` directory (and it's respective sub folder).
   * Terminal output indicates the file path for each recording.

## Usage

### Interactive Mode

The script is organized into cell blocks (marked with `# %%`), allowing you to run sections independently in an interactive IDE (e.g., Visual Studio Code).

* **Step 1:** Open `demonstration.py` in VS Code.
* **Step 2:** Execute individual cells (`# %%`) to:

  * Observe pre-trained agents on enviornment.
  * Train PPO/DQN agents.
  * Play back recorded videos.

Most cells can be run standalone in any order. However, the first two cells must always be executed to initialize the environment and dependencies. For best results, we recommend following the file’s linear execution order.

### Command-Line Mode

To run the full demonstration end-to-end and record videos, use:

```bash
python demonstration.py
```

The script will print the paths of the saved videos and also print additional information.


# Spatial Heatmap Visualization

## Overview
Two nearly identical scripts for visualizing agent trajectories as continuous spatial heat-maps in the `MediumDeliveryEnv`.  
- **dqn_spatial_heatmap.py**: uses a pre‐trained DQN agent.  
- **ppo_spatial_heatmap.py**: uses a pre‐trained PPO agent.
Both scripts share the same interface and differ only in the RL algorithm and default KDE settings.

## Usage
```bash
python dqn_spatial_heatmap.py [--model PATH] [--map NAME] [--trials N]
                             [--bw FLOAT] [--cmap NAME] [--out FILE] [-v]

python ppo_spatial_heatmap.py [--model PATH] [--map NAME] [--trials N]
                             [--bw FLOAT] [--cmap NAME] [--out FILE] [-v]
```

## Arguments
`--model`: path to your `.pth` file<br>
`--map`: one of the keys in `MAIL_DELIVERY_MAPS`<br>
`--trials`: number of episodes (default: 100)<br>
`--bw`: KDE bandwidth adjust (DQN default 0.3, PPO default 0.5)<br>
`--cmap`: matplotlib colormap (e.g. `viridis`, `magma`)<br>
`--out`: output PNG filename<br>
`-v`: verbose logging

## Output
Heat-map PNGs are saved under sheatmap_results/<out>.


# AI Disclaimer

AI assistants _claude, chatgpt_ were utilized for debugging, code refinement, optimization, and annotation. However, no code was directly copied from a language model. Instead, multiple iterative prompts were used to debug, validate, and brainstorm ideas, combining lecture materials and our own ideas. Furthermore, LLMs were also utilized to assist in typesetting latex, paraphrasing, and grammar checking various sections of the report.
