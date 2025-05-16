import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Used claude.ai to quickly iteration on and improve the function

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Used claude.ai to quickly iteration on and improve the function

def create_algorithm_comparison_plots(df, x_param, x_values, algorithms, 
                                     title, x_label, fixed_params=None,
                                     colors=None, figsize=(15, 3),
                                     use_log_scale=True, save_path=None,
                                     conv_threshold=0.01, percentile_range=75):
    """
    Create comparison plots for different algorithms across varying parameter values.
    With reduced error bands showing only up to the specified percentile range.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The data with algorithm results
    x_param : str
        The parameter name to vary on x-axis
    x_values : list
        The values of the parameter to plot
    algorithms : list
        List of algorithm names to compare
    title : str
        The main title for the figure
    x_label : str
        Label for the x-axis
    fixed_params : dict, optional
        Dictionary of parameters to hold constant {param_name: value}
    colors : list, optional
        List of colors for each algorithm
    figsize : tuple, optional
        Figure size (width, height)
    use_log_scale : bool, optional
        Whether to use log scale for convergence plot
    save_path : str, optional
        Path to save the figure
    conv_threshold : float, optional
        Threshold for convergence (delta V)
    percentile_range : int, optional
        Percentile range to show in error bands (e.g., 75 for 75th percentile)
    
    Returns:
    --------
    fig : matplotlib Figure
        The created figure object
    """
    if colors is None:
        colors = ['blue', 'red', 'green']
    
    if fixed_params is None:
        fixed_params = {}
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axis_label_fontsize = 14
    subplot_title_fontsize = 18
    
    axes[0].set_title(f"Convergence Speed (Threshold = {conv_threshold})", fontsize=subplot_title_fontsize)
    axes[0].set_xlabel(x_label, fontsize=axis_label_fontsize)
    axes[0].set_ylabel("Iterations to Converge", fontsize=axis_label_fontsize)
    if use_log_scale:
        axes[0].set_yscale('log')
    
    axes[1].set_title("Final Policy Quality", fontsize=subplot_title_fontsize)
    axes[1].set_xlabel(x_label, fontsize=axis_label_fontsize)
    axes[1].set_ylabel("Cumulative Reward", fontsize=axis_label_fontsize)
    
    # Define percentile calculation for error bands
    def get_percentile_bounds(data):        
        lower_percentile = np.percentile(data, (100 - percentile_range))
        upper_percentile = np.percentile(data, percentile_range)
        
        return lower_percentile, upper_percentile
    
    for i, alg in enumerate(algorithms):
        conv_speeds_by_param = []
        rewards_by_param = []
        conv_speeds_lower = []
        conv_speeds_upper = []
        rewards_lower = []
        rewards_upper = []
        
        for param_value in x_values:
            # Create filter conditions
            filter_conditions = {
                "algorithm": alg,
                x_param: param_value
            }
            filter_conditions.update(fixed_params)
            
            # Apply filters
            subset = df.copy()
            for key, value in filter_conditions.items():
                subset = subset[subset[key] == value]
            
            conv_speed_results = []
            reward_results = []
            
            for _, group in subset.groupby('run_id'):
                # Find first iteration/episode where convergence threshold is met
                conv_rows = group[group['conv_metricV'] <= conv_threshold]
                
                if len(conv_rows) > 0:
                    # Get the first row where threshold is met
                    first_conv_row = conv_rows.iloc[0]
                    
                    # Get the iteration/episode where convergence occurred
                    if alg == "Value Iteration":
                        conv_speed = first_conv_row['step']
                    else: 
                        conv_speed = first_conv_row['episode']
                        
                    conv_speed_results.append(conv_speed)
                else:
                    # If never converged, use max iterations/episodes
                    if alg == "Value Iteration":
                        max_iter = group['step'].max()
                    else:
                        max_iter = group['episode'].max()
                    conv_speed_results.append(max_iter)
                
                if len(group) > 0:
                    final_reward = group['cumulative_reward'].iloc[-1]
                    reward_results.append(final_reward)
            
            # Calculate statistics
            if conv_speed_results:
                conv_speeds_by_param.append(np.mean(conv_speed_results))
                lower, upper = get_percentile_bounds(conv_speed_results)
                conv_speeds_lower.append(lower)
                conv_speeds_upper.append(upper)
            else:
                conv_speeds_by_param.append(np.nan)
                conv_speeds_lower.append(np.nan)
                conv_speeds_upper.append(np.nan)
                
            if reward_results:
                rewards_by_param.append(np.mean(reward_results))
                lower, upper = get_percentile_bounds(reward_results)
                rewards_lower.append(lower)
                rewards_upper.append(upper)
            else:
                rewards_by_param.append(np.nan)
                rewards_lower.append(np.nan)
                rewards_upper.append(np.nan)
        
        color = colors[i % len(colors)]
        axes[0].plot(x_values, conv_speeds_by_param, 'o-', color=color, label=alg)
        axes[1].plot(x_values, rewards_by_param, 'o-', color=color, label=alg)
        
        # Add reduced error bands using percentile range
        axes[0].fill_between(x_values, conv_speeds_lower, conv_speeds_upper, 
                             color=color, alpha=0.2)
        
        axes[1].fill_between(x_values, rewards_lower, rewards_upper, 
                             color=color, alpha=0.2)
    
    # Place legend in the upper right of both plots
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig


df = pd.read_csv("evaluate/results_FIN_SIGMA.csv", sep=",")
algorithms = ["Value Iteration", "Monte Carlo", "Q learning"]

stochasticity_values = [0.0, 0.05, 0.1, 0.3]
fig1 = create_algorithm_comparison_plots(
    df=df,
    x_param="stochasticity",
    x_values=stochasticity_values,
    algorithms=algorithms,
    title="Effect of Stochasticity (σ) on Algorithm Performance",
    x_label="Stochasticity (σ)",
    fixed_params={"discount_factor": 0.99},
    save_path="evaluate/sigma_comparison.svg"
)

df = pd.read_csv("evaluate/results_FIN_GAMMA.csv", sep=",")
algorithms = ["Value Iteration", "Monte Carlo", "Q learning"]

gamma_values = [0.6, 0.8, 0.9, 0.95, 0.99]
fig2 = create_algorithm_comparison_plots(
    df=df,
    x_param="discount_factor",
    x_values=gamma_values,
    algorithms=algorithms,
    title="Effect of Discount Factor (γ) on Algorithm Performance",
    x_label="Discount Factor (γ)",
    fixed_params={"stochasticity": 0.1},
    save_path="evaluate/gamma_comparison.svg"
)

df = pd.read_csv("evaluate/results_FIN_ALPHA.csv", sep=",")
algorithms = ["Monte Carlo", "Q learning"]

initial_alpha = [0.01, 0.1, 0.5, 1, 2]

fig1 = create_algorithm_comparison_plots(
    df=df,
    x_param="initial_alpha",
    x_values=initial_alpha,
    algorithms=algorithms,
    title="Effect of Initial Learning Rate (α₀) on Performance",
    x_label="Initial Learning Rate (α₀)",
    fixed_params=None,
    save_path="evaluate/alpha_comparison.svg",
    use_log_scale=True
)

df = pd.read_csv("evaluate/results_FIN_EPSILON.csv", sep=",")
algorithms = ["Monte Carlo", "Q learning"]

epsilon_decay = [0.9, 0.95, 0.99, 0.999, 0.9999]

fig1 = create_algorithm_comparison_plots(
    df=df,
    x_param="epsilon_decay",
    x_values=epsilon_decay,
    algorithms=algorithms,
    title="Effect of Epsilon Decay Rate (ε-decay) on Performance",
    x_label="Epsilon Decay Rate (per episode)",
    fixed_params=None,
    save_path="evaluate/epsilon_comparison.svg",
    use_log_scale=True
)

df = pd.read_csv("evaluate/results_FIN_MAX_STEPS.csv", sep=",")
algorithms = ["Monte Carlo", "Q learning"]

max_steps_per_episode = [100, 250, 500, 1000]

fig1 = create_algorithm_comparison_plots(
    df=df,
    x_param="episode_length_mc",
    x_values=max_steps_per_episode,
    algorithms=algorithms,
    title="Effect of Epsilon Decay Rate (ε-decay) on Performance",
    x_label="Maximum Steps per Episode",
    fixed_params=None,
    save_path="evaluate/episode_length_comparison.svg",
    use_log_scale=True
)
