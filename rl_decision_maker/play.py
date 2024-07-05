import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class DisasterManagementEnv(gym.Env):
    def __init__(self, initial_budget=50):
        super(DisasterManagementEnv, self).__init__()
        
        # Define action and observation space
        # Actions: 0 = Identify humanitarian info, 1 = Monitor situation, 2 = Deploy without data,
        # 3 = Don't do anything, 4 = Gather additional data, 5 = Deploy response without data
        self.action_space = spaces.Discrete(6)
        
        # Observation space: [current_state, remaining_budget, label_index]
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)
        
        self.initial_budget = initial_budget
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.current_state = 0  # Initial state
        self.remaining_budget = self.initial_budget
        self.current_label = "informative"
        self.label_dict = {
            0: "informative", 1: "non-informative",
            2: "affected individuals", 3: "infrastructure and utility damage",
            4: "other information", 5: "rescue information",
            6: "little or no damage", 7: "severe damage",
            8: "damage", 9: "no damage"
        }
        self.label_index = 0
        return np.array([self.current_state, self.remaining_budget, self.label_index], dtype=np.float32), {}

    def step(self, action):
        reward = 0
        cost = 0
        next_state = self.current_state
        done = False
        
        if self.current_state == 0:
            next_state = 1
            self.current_label = random.choice(["informative", "non-informative"])
            self.label_index = 0 if self.current_label == "informative" else 1

        elif self.current_state == 1:
            if self.current_label == "non-informative" and action == 0:
                reward = -20  # Penalty for trying to identify humanitarian info on non-informative data
            elif action == 0:
                next_state = 2
                reward = 10
                self.current_label = random.choice([
                    "affected individuals", "infrastructure and utility damage", 
                    "other information", "rescue information"])
                self.label_index = random.choice([2, 3, 4, 5])
            elif action == 4:
                next_state = 2
                reward = 5
                cost = 5  # Cost for gathering additional data
                self.current_label = random.choice(["little or no damage", "severe damage"])
                self.label_index = 6 if self.current_label == "little or no damage" else 7
            else:
                reward = -10  # Penalty for other actions

        elif self.current_state == 2:
            if action == 0 or action == 4:
                next_state = 3
                reward = 10
                cost = 5  # Cost for gathering additional data
                self.current_label = random.choice(["damage", "no damage"])
                self.label_index = 8 if self.current_label == "damage" else 9
            else:
                reward = -10

        elif self.current_state == 3:
            if action == 0 or action == 4:
                next_state = 4
                reward = 10
                cost = 5  # Cost for gathering additional data
                self.current_label = random.choice(["damage", "no damage"])
                self.label_index = 8 if self.current_label == "damage" else 9
            else:
                reward = -10

        elif self.current_state == 4:
            if self.current_label == "no damage" and action == 0:
                reward = -20  # Penalty for deploying resources when no damage
            elif action == 0:
                reward = 20  # Successful deployment
            else:
                reward = -10

        self.remaining_budget -= cost
        done = self.remaining_budget < 0 or next_state == 4

        self.current_state = next_state
        truncated = done  # Define `truncated` as the same as `done` for simplicity
        return np.array([self.current_state, self.remaining_budget, self.label_index], dtype=np.float32), reward, done, truncated, {}

    def render(self, mode='human'):
        pass  # Not needed for this simulation

    def close(self):
        pass  # Not needed for this simulation

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Create the environment
env = DisasterManagementEnv()

# Verify that the environment follows the Gymnasium interface
check_env(env)

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_disaster_management")

# Load the model
model = PPO.load("ppo_disaster_management")

# Simulate the game
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Current State: {obs[0]}, Remaining Budget: {obs[1]}, Label Index: {obs[2]}, Reward: {reward}")
    
    if done:
        if obs[1] < 0:
            print("Game Over: Budget Exhausted")
        else:
            print("Game Over: Mission Accomplished")
