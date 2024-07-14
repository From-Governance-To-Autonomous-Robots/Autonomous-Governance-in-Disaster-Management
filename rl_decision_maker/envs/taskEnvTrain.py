import gym
from gym import spaces
import numpy as np
import yaml
import pandas as pd 
import os 
import random
import pdb
from utils.utils import load_config

class TaskSequenceTrainEnv(gym.Env):
    def __init__(self,render_mode:str='human'):
        super(TaskSequenceTrainEnv, self).__init__()
        self.CONFIG = load_config()
        self.tasks = self.CONFIG['tasks']
        # Load the Datasets for each task
        self.info_dataset = self._get_data_based_on_task("info")
        self.human_dataset = self._get_data_based_on_task("human")
        self.damage_dataset = self._get_data_based_on_task("damage")
        self.satellite_dataset = self._get_data_based_on_task("satellite")
        self.drone_dataset = self._get_data_based_on_task("drone")
        # Store the Seen Indexes of the Records for each Dataset 
        self.seen_info = []
        self.seen_human = []
        self.seen_damage = []
        self.seen_satellite = []
        self.seen_drone = []
        
        self.credits_info = 5
        self.credits_human = 5
        self.credits_damage = 5
        self.credits_satellite = 5
        self.credits_drone = 5
        
        self.isTreeCorrectlyAnswered = False
        self.currentEpisode = 0
        self.currentStep = 0
        self.tree_counter = 0
        self.correct_answered_tree_counter = 0
        self.wrongly_answered_tree_counter = 0
        
        self.task_index = 0
        self.current_task_info = None
        self.ground_truth = None
        self.tree_score = 0

        # Define action space
        self.action_space = spaces.Discrete(5)  # Actions 0, 1, 2, 3, 4
        
        # Define observation space
        max_length = max([len(self.info_dataset.iloc[0]["prediction_conf"]),
                          len(self.human_dataset.iloc[0]["prediction_conf"]),
                          len(self.damage_dataset.iloc[0]["prediction_conf"]),
                          len(self.satellite_dataset.iloc[0]["prediction_conf"]),
                          len(self.drone_dataset.iloc[0]["prediction_conf"])])
        self.observation_space = spaces.Box(low=0, high=1, shape=(max_length + len(self.tasks),), dtype=np.float32)
        self.reset()
        
    def _random_select_record_from_dataset(self, task: str):
        
        if task == "info":
            dataset = self.info_dataset
        elif task == "human":
            dataset = self.human_dataset
        elif task == "damage":
            dataset = self.damage_dataset
        elif task == "satellite":
            dataset = self.satellite_dataset
        elif task == "drone":
            dataset = self.drone_dataset
        else:
            dataset = self.info_dataset
        
        if dataset.empty:
            raise ValueError("The dataset is empty")

        if task == "info":
            remaining_indices = list(set(dataset.index) - set(self.seen_info))
        elif task == "human":
            remaining_indices = list(set(dataset.index) - set(self.seen_human))
        elif task == "damage":
            remaining_indices = list(set(dataset.index) - set(self.seen_damage))
        elif task == "satellite":
            remaining_indices = list(set(dataset.index) - set(self.seen_satellite))
        elif task == "drone":
            remaining_indices = list(set(dataset.index) - set(self.seen_drone))
        else:
            remaining_indices = list(set(dataset.index) - set(self.seen_info))

        if not remaining_indices:
            raise ValueError("All records have been seen")

        selected_idx = random.choice(remaining_indices)
        
        if task == "info":
            self.seen_info.append(selected_idx)
        elif task == "human":
            self.seen_human.append(selected_idx)
        elif task == "damage":
            self.seen_damage.append(selected_idx)
        elif task == "satellite":
            self.seen_satellite.append(selected_idx)
        elif task == "drone":
            self.seen_drone.append(selected_idx)
        else:
            self.seen_info.append(selected_idx)
        
        return dataset.loc[selected_idx]["ground_truth"], dataset.loc[selected_idx]["prediction_conf"]

    def _handle_gather_additional_data(self,task):
        if task == "info":
            dataset = self.info_dataset
        elif task == "human":
            dataset = self.human_dataset
        elif task == "damage":
            dataset = self.damage_dataset
        elif task == "satellite":
            dataset = self.satellite_dataset
        elif task == "drone":
            dataset = self.drone_dataset
        else:
            dataset = self.info_dataset
        
        dataset = dataset[dataset["ground_truth"] == self.ground_truth]
        
        if dataset.empty:
            raise ValueError("The dataset is empty")

        if task == "info":
            remaining_indices = list(set(dataset.index) - set(self.seen_info))
        elif task == "human":
            remaining_indices = list(set(dataset.index) - set(self.seen_human))
        elif task == "damage":
            remaining_indices = list(set(dataset.index) - set(self.seen_damage))
        elif task == "satellite":
            remaining_indices = list(set(dataset.index) - set(self.seen_satellite))
        elif task == "drone":
            remaining_indices = list(set(dataset.index) - set(self.seen_drone))
        else:
            remaining_indices = list(set(dataset.index) - set(self.seen_info))

        if not remaining_indices:
            raise ValueError("All records have been seen")

        selected_idx = random.choice(remaining_indices)
        if task == "info":
            self.seen_info.append(selected_idx)
        elif task == "human":
            self.seen_human.append(selected_idx)
        elif task == "damage":
            self.seen_damage.append(selected_idx)
        elif task == "satellite":
            self.seen_satellite.append(selected_idx)
        elif task == "drone":
            self.seen_drone.append(selected_idx)
        else:
            self.seen_info.append(selected_idx)
        
        return dataset.loc[selected_idx]["ground_truth"], dataset.loc[selected_idx]["prediction_conf"]
    
    def _get_data_based_on_task(self, task: str):
        dataset = pd.read_csv(os.path.join(self.CONFIG['data_path'], task, f"train_inference_results.csv"))
        dataset["prediction_conf"] = dataset["prediction_conf"].apply(lambda x:list(map(float, x.strip("[]").split())))
        return dataset
    
    def reset(self):
        self.seen_info = []
        self.seen_human = []
        self.seen_damage = []
        self.seen_satellite = []
        self.seen_drone = []
        
        self.credits_info = 5
        self.credits_human = 5
        self.credits_damage = 5
        self.credits_satellite = 5
        self.credits_drone = 5
        
        self.isTreeCorrectlyAnswered = False
        self.number_of_times_additional_data_requested = 0
        self.currentStep = 0
        
        self.task_index = 0
        self.tree_score = 0
        self.ground_truth, self.current_task_info = self.get_task_data()
        obs = self._get_observation()
        return obs

    def get_task_data(self):
        task = self.tasks[self.task_index]
        return self._random_select_record_from_dataset(task)
    
    def _get_observation(self):
        task_vector = np.zeros(len(self.tasks))
        task_vector[self.task_index] = 1
        task_info = self.current_task_info
        observation = np.concatenate([task_vector, task_info])
        if observation.shape[0] < 9:
            observation = np.concatenate([observation, np.zeros(9 - observation.shape[0])])
        return observation

    def step(self, action):
        reward = 0
        done = False
        
        if action == 4:
            task = self.tasks[self.task_index]
            if task == "info":
                if self.credits_info != 0:
                    self.credits_info -= 1
                    self.ground_truth, self.current_task_info = self._handle_gather_additional_data(task)
                    reward = -1
                    self.number_of_times_additional_data_requested += 1
                    
            elif task == "human":
                if self.credits_human != 0:
                    self.credits_human -= 1
                    self.ground_truth, self.current_task_info = self._handle_gather_additional_data(task)
                    self.number_of_times_additional_data_requested += 1
                    reward = -1
            
            elif task == "damage":
                if self.credits_damage != 0:
                    self.credits_damage -= 1
                    self.ground_truth, self.current_task_info = self._handle_gather_additional_data(task)
                    self.number_of_times_additional_data_requested += 1
                    reward = -1
            
            elif task == "satellite":
                if self.credits_satellite != 0:
                    self.credits_satellite -= 1
                    self.ground_truth, self.current_task_info = self._handle_gather_additional_data(task)
                    self.number_of_times_additional_data_requested += 1
                    reward = -1
            
            elif task == "drone":
                if self.credits_drone != 0:
                    self.credits_drone -= 1
                    self.ground_truth, self.current_task_info = self._handle_gather_additional_data(task)
                    self.number_of_times_additional_data_requested += 1
                    reward = -1
            
        elif action == self.ground_truth:
            reward = 1
            self.task_index += 1
            self.correct_answered_tree_counter +=1
            self.isTreeCorrectlyAnswered = True
        
        elif action != self.ground_truth:
            reward = -5
            self.task_index += 1
            self.wrongly_answered_tree_counter +=1
            self.isTreeCorrectlyAnswered = False
        
        if self.task_index >= len(self.tasks):
            self.tree_counter += 1
            done = True
            self.task_index = 0
            task = self.tasks[self.task_index]
            self.ground_truth, self.current_task_info = self._random_select_record_from_dataset(task)
        else:
            task = self.tasks[self.task_index]
            self.ground_truth, self.current_task_info = self._random_select_record_from_dataset(task)

        self.tree_score += reward
        
        if done:
            self.currentEpisode += 1
            info = {
                "episode_ended":True,
                "tree_score": self.tree_score,
                "isTreeCorrectlyAnswered":self.isTreeCorrectlyAnswered,
                "currentEpisode":self.currentEpisode,
                "currentStep":self.currentStep,
                "tree_id":self.tree_counter,
                "currentStepReward":reward,
                "number_of_correctly_answered_trees":self.correct_answered_tree_counter,
                "number_of_wrongly_answered_trees":self.wrongly_answered_tree_counter,
                "number_of_times_additional_data_requested":self.number_of_times_additional_data_requested
            }
        else:
            info = {
                "episode_ended":False,
                "tree_score": self.tree_score,
                "isTreeCorrectlyAnswered":self.isTreeCorrectlyAnswered,
                "currentEpisode":self.currentEpisode,
                "currentStep":self.currentStep,
                "tree_id":self.tree_counter,
                "currentStepReward":reward,
                "number_of_correctly_answered_trees":self.correct_answered_tree_counter,
                "number_of_wrongly_answered_trees":self.wrongly_answered_tree_counter,
                "number_of_times_additional_data_requested":self.number_of_times_additional_data_requested
            }
            
        obs = self._get_observation()
        return obs, reward, done, info 

    def render(self, mode='human'):
        pass

    def close(self):
        pass