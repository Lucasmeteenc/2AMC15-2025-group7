In this assignment we are asked to implement a variety of reinforcement learning algorithms that are going to assist us in identifying the optimal policy a delivery robot should follow.

The algorithms that were implemented are:
1. Value Iteration
2. On-policy Monte Carlo 
3. Q-learning

### Agents location

2AMC15-2025-group/
├── agents/ # Contains implementations of different RL agents..
│ ├── init.py..
│ ├── base_agent.py..
│ ├── `mc_agent.py` # On-Policy Monte Carlo Control agent..
│ ├── `vi_agent.py` # Value Iteration agent..
│ ├── `q_learning_agent.py` # Q-Learning agent using ε-greedy strategy..
│ ├── random_agent.py..
│ ├── null_agent.py..


Example run with default values

# Run Value Iteration
python train.py ./grid_configs/A1_grid.npy --agent=vi --no_gui

# Run Q-learning
python train.py ./grid_configs/A1_grid.npy --agent=ql --no_gui

# Run Monte Carlo
python train.py ./grid_configs/A1_grid.npy --agent=mc --no_gui

Below is a list of command-line options supported by `train.py`.


### General Arguments

| Argument                  | Type     | Default | Description |
|---------------------------|----------|---------|-------------|
| `GRID`                    | `Path`   | –       | Path(s) to the grid configuration file(s). Can specify multiple. |
| `--no_gui`                | flag     | False   | Disables GUI rendering to speed up training. |
| `--sigma`                 | `float`  | 0.1     | Controls stochasticity of the environment. |
| `--fps`                   | `int`    | 30      | Frames per second for rendering (ignored if `--no_gui` is set). |
| `--iters`                 | `int`    | 50000   | Number of iterations (VI) or episodes (MC/QL). |
| `--random_seed`           | `int`    | 0       | Random seed for reproducibility. |
| `--gamma`                 | `float`  | 0.99    | Discount factor for future rewards. |

### Agent selection

| Argument                  | Type     | Default | Description |
|---------------------------|----------|---------|-------------|
| `--agent`                 | `str`    | "vi"    | Choose the agent: `vi` (Value Iteration), `mc` (Monte Carlo), or `ql` (Q-learning). |

### Monte Carlo and Q-learning shared Parameters

| Argument                       | Type     | Default | Description |
|--------------------------------|----------|---------|-------------|
| `--max_steps_per_episode`      | `int`    | 5000    | Max steps allowed per episode (safety cap). |
| `--epsilon`                    | `float`  | 1.0     | Initial exploration rate. |
| `--min_epsilon`                | `float`  | 0.0005  | Minimum exploration rate. |
| `--epsilon_decay`              | `float`  | 0.999   | Epsilon decay rate per episode. |

### Specific parameters

| Argument                       | Type     | Default | Description |
|--------------------------------|----------|---------|-------------|
| `--early_stopping_patience_mc` | `int`    | 1000    | Stop if policy remains unchanged for this many episodes.|
| `--early_stopping_patience_ql` | `int`    | 50      | Stop if policy remains unchanged for this many episodes.|

### Experiments reproduction

The script that performs the experiments is plots.py, located in the evaluate folder.
To run it, the only required files are the results files: results_FIN_ALPHA.csv, results_FIN_EPSILON.csv, results_FIN_GAMMA.csv, resutls_FIN_MAX_STEPS.csv, results_FIN_SIGMA.csv, which contain the findings of our experiments.

### AI Disclaimer

AI tools were used only to assist with:
1. Debugging code-related issues
2. Generating code snippets for graph visualization and plotting
