"""Agent Base.

We define the base class for all agents in this file.
"""

from abc import ABC, abstractmethod

import uuid
import numpy as np
import os
import csv

class BaseAgent():
    def __init__(self):
        """Base agent. All other agents should build on this class.
        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """
        self.run_id = uuid.uuid4()
        self.algorithm = None
        self.stochasticity = -1.0
        self.discount_factor = -1.0
        self.episode_length_mc = -1
        self.grid_name = None
        self.reward_function = "Default"
        self.cumulative_reward = -1.0
        self.step = 0
        self.episode = 0
        self.logging_file_path = "evaluate/results.csv"

    def log_metrics(self, cumulative_reward,alpha=-1.0, epsilon=-1.0, conv_metricV=-1.0,conv_metricQ=-1.0):
        if not hasattr(self, 'run_id'):
            raise ValueError("BaseAgent has not been initialized. Ensure, you run 'super().__init__()'")

        if self.algorithm is None:
            raise ValueError(
                "You need to set the parameters for the agent in order to log the result.\nCall the function: set_parameters"
            )
        
        file_path = self.logging_file_path
        
        # Header formatting generated using ChatGPT:
        # https://chatgpt.com/share/681dc499-9e24-8001-93b2-1d11e7a01f58
        # This is modified to fit our needs
        
        # Define the log data
        log_data = [
            str(self.run_id),
            self.algorithm,
            self.stochasticity,
            self.discount_factor,
            alpha,
            epsilon,
            self.episode_length_mc,
            self.grid_name,
            self.reward_function,
            cumulative_reward,
            self.step,
            self.episode,
            conv_metricV,
            conv_metricQ
        ]

        # Check if the file exists
        file_exists = os.path.exists(file_path)

        # Open file in append mode and write log data
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # If the file doesn't exist, write the header first
            if not file_exists:
                header = [
                    'run_id', 'algorithm', 'stochasticity', 'discount_factor', 'learning_rate',
                    'epsilon', 'episode_length_mc', 'grid_name', 'reward_function', 
                    'cumulative_reward', 'step', 'episode', 'conv_metricV','conv_metricQ'
                ]
                writer.writerow(header)

            # Append the current log data
            writer.writerow(log_data)
            

    def _set_parameters(
        self,
        algorithm: str,
        stochasticity: float,
        discount_factor: float,
        grid_name: str,
        episode_length_mc: int = -1,
        reward_function="Default",
        logging_file_path:str = None,
    ):
        """Sets the parameters for logging. If a value is not relevant. Set it to -1.0 for floats, or None for strings

        Args:
            algorithm (str): _description_
            stochasticity (float): _description_
            discount_factor (float): _description_
            grid_name (str): _description_
            learning_rate (float, optional): _description_. Defaults to -1.0.
            epsilon (float, optional): _description_. Defaults to -1.0.
            episode_length_mc (int, optional): _description_. Defaults to -1.
            reward_function (str, optional): _description_. Defaults to "Default".

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        if algorithm == "Monte Carlo" and episode_length_mc < 0:
                raise ValueError(
                    "Not all parameters of Monte Carlo have been provided.\nGive episode_length_mc"
                )

        # From here on out, we assume parameters have been set correctly
        self.run_id = uuid.uuid4()
        self.algorithm = algorithm
        self.stochasticity = stochasticity
        self.discount_factor = discount_factor
        self.grid_name = grid_name
        self.episode_length_mc = episode_length_mc
        self.reward_function = reward_function
        
        if logging_file_path:
            self.logging_file_path = logging_file_path

    @abstractmethod
    def take_action(self, state: tuple[int, int]) -> int:
        """Any code that does the action should be included here.

        Args:
            state: The updated position of the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, state: tuple[int, int], reward: float, action: int):
        """Any code that processes a reward given the state and updates the agent.

        Args:
            state: The updated position of the agent.
            reward: The value which is returned by the environment as a
                reward.
            action: The action which was taken by the agent.
        """
        raise NotImplementedError
    
