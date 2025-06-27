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


# PPO Agent

The Proximal Policy Optimization (PPO) implementation is structured across the following files:

| File Name         | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
|  `ppo_network.py`  | Defines the actor and critic neural networks used in PPO.                   |
| `ppo_utils.py`    | Provides utility functions such as Generalized Advantage Estimation (GAE) and logging. |
| `ppo_agent.py`    | Implements the PPO algorithm and the training process.                         |

## PPO Training

### Script: `ppo_agent.py`

The script: 
- Parses training arguments
- Sets random seeds and training environment
- Initializes the PPO agent
- Trains the agent
- Saves the logs and model parameters

### PPO Training Parameters

| Argument                | Type    | Default        | Description |
|-------------------------|---------|----------------|-------------|
| `--num-envs`            | `int`   | `8`            | Number of parallel environments used during training. |
| `--horizon`             | `int`   | `2_048`         | Number of steps per environment before each PPO update. |
| `--total-timesteps`     | `int`   | `2_000_000`    | Total number of environment steps to train for. |
| `--hidden-dim`          | `int`   | `128`           | Width of each fully connected hidden layer in the networks. |
| `--learning-rate`       | `float` | `1e-3`       | Learning rate for the optimizer. |
| `--batch-size`          | `int`   | `64`           | Minibatch size for PPO updates. |
| `--n-epochs`            | `int`   | `10`           | Number of training epochs per PPO update. |
| `--clip-range`          | `float` | `0.1`          | Clipping range (epsilon) for the PPO objective. |
| `--gamma`               | `float` | `0.99`         | Discount factor for future rewards. |
| `--gae-lambda`          | `float` | `0.9`         | Lambda parameter for Generalized Advantage Estimation (GAE). |
| `--value-coef`          | `float` | `1`          | Weight of value function loss in the total loss. |
| `--entropy-coef`        | `float` | `0.0`         | Weight of entropy bonus to encourage exploration. |
| `--max-grad-norm`       | `float` | `0.5`          | Maximum gradient norm for clipping (stabilizes training). |
| `--use-gae`             | `bool`  | `True`         | Whether to use GAE for advantage estimation. |
| `--seed`                | `int`   | `0`           | Random seed for reproducibility. |
| `--log-interval`        | `int`   | `1`           | Log training metrics every N updates. |
| `--checkpoint-interval` | `int`   | `1`          | Save model checkpoint every N updates. |
| `--log-window`          | `int`   | `100`           | Rolling window size for smoothing log metrics. |
| `--log-dir`             | `str`   | `"logs"`      | Directory to save training logs. |
| `--checkpoint-dir`      | `str`   | `"checkpoints_ppo"` | Directory to save model checkpoints. |
| `--video-dir`           | `str`   | `"videos"`    | Directory to store evaluation videos. |
| `--map-name`            | `str`   | `"default"`  | Environment map name for the training scenario. Other values "inside", "empty".|

The default parameters were selected based on the best-performing results from the hyperparameter sweeps.

### Example run: PPO Training using the best identified hyperparameters
- python ppo_agent.py


## PPO Testing

### Script: `ppo_test.py`

The script: 
- Loads a saved model
- Runs multiple test episodes for the specified environment
- Records and saves videos

### PPO Test Parameters

The parameters are set and can be modified inside the 'main()' function.

| Parameter   | Type   | Default                                      | Description                                         |
|-------------|--------|----------------------------------------------|-----------------------------------------------------|
| `map`       | string | `"default"`                                  | Environment map name for the training scenario. Other values "inside", "empty" |
| `trials`    | int    | `10`                                         | Number of test episodes to run.                      |
| `model_path`| string | `"checkpoints_ppo/final_model_default.pth"` | Path to the saved PPO model checkpoint file.        |
| `video_dir` | string | `"videos/test_videos_ppo_{map}"`           | Directory where test videos will be saved.           |

# DQN Agent

The Deep Q-Learning (DQN) implementation is structured across the following files:

| File Name         | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
|  `dqn_model.py`  | Defines the DQN architecture.                 |
| `dqn_agent.py`    | Implements the DQN algorithm and the training process.                         |

## DQN Training

### Script: `dqn_agent.py`

The script: 
- Parses training arguments
- Sets random seeds and training environment
- Initializes the DQN agent
- Trains the agent
- Saves the logs and model parameters

| Argument             | Type   | Default Value           | Description                                      |
|----------------------|--------|--------------------------|--------------------------------------------------|
| `--total-episodes`   | `int`  | `15_000` | Total number of training episodes.              |
| `--batch-size`       | `int`  | `64`     | Batch size for sampling from replay memory.     |
| `--memory-size`      | `int`  | `200_000`    | Maximum size of the replay memory buffer.       |
| `--learning-rate`    | `float`| `0.0001`  | Learning rate for the optimizer.                |
| `--gamma`            | `float`| `0.99`          | Discount factor for future rewards.             |
| `--epsilon-start`    | `float`| `1.0`  | Initial epsilon value for exploration.          |
| `--epsilon-end`      | `float`| `0.01`    | Final epsilon value after decay.                |
| `--epsilon-decay`    | `float`| `0.999`  | Rate of epsilon decay per episode.              |
| `--hidden-dim`       | `int`  | `256`     | Size of hidden layers in the Q-network.         |
| `--checkpoint-dir`   | `str`  | `"checkpoints_dqn"` | Directory to save model checkpoints.            |
| `--log-interval`     | `int`  | `10`   | Interval (in episodes) for logging stats.       |
| `--seed`             | `int`  | `0`           | Random seed for reproducibility.                |
| `--map-name`         | `str`  | `default`       | Environment map name for the training scenario. Other values "inside", "empty".  |

The default parameters were selected based on the best-performing results from the hyperparameter sweeps.

### Example run: DQN Training using the best identified hyperparameters
- python dqn_agent.py


## DQN Testing

### Script: `dqn_test.py`

The script: 
- Loads a saved model
- Runs multiple test episodes for the specified environment
- Records and saves videos

### DQN Test Parameters

The parameters are set and can be modified inside the 'main()' function.

| Parameter   | Type   | Default                                      | Description                                         |
|-------------|--------|----------------------------------------------|-----------------------------------------------------|
| `map`       | string | `"default"`                                  | Environment map name for the training scenario. Other values "inside", "empty" |
| `trials`    | int    | `50`                                         | Number of test episodes to run.                      |
| `model_path`| string | `"checkpoints_dqn/final_model_default.pth"` | Path to the saved DQN model checkpoint file.        |
| `video_dir` | string | `"videos/test_videos_dqn_{map}"`           | Directory where test videos will be saved.           |

# Ablation study 

The ablation study explored the algorithms' performance on two maps, the `default` and the `inside`, using either 1 or 10 rays.

In order to reproduce the results: 
- modify the files `dqn_agent.py` and `ppo_agent.py` by trying both `inside` and `default` values of the `--map-name` parameter. 
- modify the file `medium_delivery_env.py` by changing the `nr_rays` variable from `10` to `1`.

1. Change the map used for training:
   - Modify the `--map-name` argument in both `dqn_agent.py` and `ppo_agent.py`:
     ```bash
     --map-name='default'
     --map-name='inside'
     ```

2. Change the number of rays used in the environment:
   - Open the file `medium_delivery_env.py`
   - Locate the `nr_rays` variable and update it:
     ```python
     nr_rays = 1  # Change from 10 to 1
     ```

Run the training scripts with all combinations of:
- `--map-name` values: `default`, `inside`
- `nr_rays`: `10` and `1`


# Sweeps

The scripts `ppo_sweep.py` and `dqn_sweep.py` perform hyperparameter sweeps for the agents using Weights & Biases.

# Environment

The environment used for assessing the algorithms' performance is `medium_delivery_env.py` located in the `environments` folder. Specifically, the environment simulates a delivery robot in a continuous state space that has to pick up and deliver a package while avoiding obstacles using discrete actions and lidar-based perception.

# AI Disclaimer

AI assistants _claude, chatgpt_ were utilized for debugging, code refinement, optimization, and annotation. However, no code was directly copied from a language model. Instead, multiple iterative prompts were used to debug, validate, and brainstorm ideas, combining lecture materials and our own ideas. Furthermore, LLMs were also utilized to assist in typesetting latex, paraphrasing, and grammar checking various sections of the report.
