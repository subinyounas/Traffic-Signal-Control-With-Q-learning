import gym
from gym import spaces
import numpy as np
from simulation import Main


class TrafficSignalEnv(gym.Env):
    def __init__(self):
        super(TrafficSignalEnv, self).__init__()

        # Define observation space
        self.observation_space = spaces.MultiDiscrete(
            [11, 11]
        )  # number of cars in each direction (0-10)

        # Define action space
        self.action_space = spaces.Discrete(
            2
        )  # increase/decrease duration of NS green light

        # Initialize simulation parameters
        self.signals = [10, 10]  # NS green light duration, EW green light duration
        self.waiting_times = {0: 0, 1: 0}  # Waiting times for NS and EW directions
        self.current_state = (0, 0)  # Initial state

    def step(self, action):
        # Apply action
        if action == 0:  # Increase NS green light duration
            self.signals[0] += 1
            self.signals[1] -= 1
        elif action == 1:  # Decrease NS green light duration
            self.signals[0] -= 1
            self.signals[1] += 1

        # Simulate traffic and calculate reward
        self.simulate_traffic()
        reward = -sum(self.waiting_times.values())

        # Update state
        self.current_state = tuple(self.signals)

        # Return observation, reward, done, info
        return self.current_state, reward, False, {}

    def reset(self):
        # Reset simulation to initial state
        self.signals = [10, 10]
        self.waiting_times = {0: 0, 1: 0}
        self.current_state = (0, 0)
        return self.current_state

    def close(self):
        # Optional: clean up resources
        pass

    def simulate_traffic(self):
        # Simulate 
        pass
