"""Q-learning Agent.

This is an agent that ...
"""
from random import randint
import numpy as np

from agents import BaseAgent
from world import Environment
from tqdm import trange


class QLearningAgent(BaseAgent):
    def __init__(self, grid: np.ndarray, grid_name: str, gamma: float, nr_actions: int = 4, stochasticity=-1, initial_epsilon = 1.0, max_steps_per_episode=-1, reward_function="Default"):
        """...

        ...
        """
        super().__init__()

        self.nr_actions = nr_actions

        self.n_cols,self.n_rows = grid.shape

        self.Q_table = np.zeros((self.n_rows, self.n_cols, self.nr_actions))
        self.Q_table_old = np.zeros((self.n_rows, self.n_cols, self.nr_actions))
        self.V_old = np.max(self.Q_table,axis=2)

        self.gamma = gamma

        self.alpha = 0.5
        self.epsilon = initial_epsilon

        self.old_state = None
        
        # Early exit if td improvement is too little
        self.little_improvement_steps = 0

        self._set_parameters("Q learning", stochasticity=stochasticity, discount_factor=gamma, grid_name=grid_name, episode_length_mc=max_steps_per_episode, reward_function=reward_function)

    def decay_learning_params(self, nEpisodes: int, episode: int):
        if episode > 0.3*nEpisodes:
            self.alpha = self.alpha * 0.9995      #initial_alpha / (1 + i / 1000)
            self.epsilon = self.epsilon * 0.999  #max(min_epsilon, initial_epsilon * np.exp(-i / decay_rate))

    def update(self, state: tuple[int, int], reward: float, action):
        """Any code that processes a reward given the state and updates the agent.

        Args:
            state: The updated position of the agent.
            reward: The value which is returned by the environment as a
                reward.
            action: The action which was taken by the agent.
        """
        # TD error
        TD_error = reward + self.gamma * np.max(self.Q_table[state[0],state[1]]) - self.Q_table[self.old_state[0], self.old_state[1], action]

        # Q-learning update
        self.Q_table[self.old_state[0], self.old_state[1], action] = self.Q_table[self.old_state[0], self.old_state[1], action] + self.alpha * TD_error
        
        if abs(self.alpha * TD_error) < 1e-6:
            self.little_improvement_steps += 1
        else:
            self.little_improvement_steps = 0

    def take_action(self, state: tuple[int, int], evaluate: bool = False) -> int:
        """Any code that does the action should be included here.

        Args:
            state: The updated position of the agent.
        """
        self.old_state = state

        # Epsilon greedy
        if evaluate or np.random.random() > self.epsilon:
            return np.argmax(self.Q_table[state[0], state[1]])
        else:
            return np.random.randint(self.nr_actions)
        
    def train(self, env: Environment, num_episodes: int, iters: int, early_stopping_patience: int):
        
        for episode in trange(num_episodes):
            
            state = env.reset()
            self.decay_learning_params(num_episodes,episode)

            self.Q_table_old = np.copy(self.Q_table)

            for _ in range(iters):
                
                # Agent takes an action based on the latest observation and info.
                action = self.take_action(state)

                # The action is performed in the environment
                state, reward, terminated, info = env.step(action)
                
                # If the final state is reached, stop.
                if terminated:
                    break

                self.update(state, reward, info["actual_action"])

                # Every 1000 steps write an iter log
                if self.step % 1000 == 0:
                    pass
                    # self.log_metrics(env.world_stats["cumulative_reward"], self.alpha, self.epsilon)
                self.step += 1
                
            

            # if self.episode % 100 == 0:
            conv_metricV,conv_metricQ = self.get_convergence_metric()
            self.log_metrics(env.world_stats["cumulative_reward"], self.alpha, self.epsilon,conv_metricV,conv_metricQ)
            self.episode += 1

            if self.little_improvement_steps > early_stopping_patience:
                print(f"Early exit after {episode} episodes.")
                break

        print(f"{np.argmax(self.Q_table, axis=2)=}")


    def get_convergence_metric(self):
        """
        Computes value functions for current and previous Q-tables and returns absolute
        difference between current and previous value function.
        """
        
        V = np.max(self.Q_table, axis=2) #compute current value function
        max_diff_V = np.max(np.abs(V - self.V_old)) #compute max difference between value functions
        self.V_old = V #update old value function for next iteration

        abs_diff = np.abs(self.Q_table - self.Q_table_old)
        max_diff_Q = np.max(abs_diff)

        return max_diff_V, max_diff_Q