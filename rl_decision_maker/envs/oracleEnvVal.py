import gym
from gym import spaces
import numpy as np
import yaml
import pandas as pd 
import os 
import random
import pdb
from utils.utils import load_config

class OracleSequenceValEnv():
    def __init__(self):
        super(OracleSequenceValEnv, self).__init__()
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
        
        self.isTreeCorrectlyAnswered = [False] * len(self.tasks)
        self.currentEpisode = 0
        self.currentStep = 0
        self.tree_counter = 0
        self.correct_answered_tree_counter = 0
        self.wrongly_answered_tree_counter = 0
        
        self.task_index = 0
        self.current_task_info = None
        self.ground_truth = None
        self.tree_score = 0

        
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
        
        return dataset.loc[selected_idx]["ground_truth"], list(dataset.loc[selected_idx]["prediction_conf"]).index(max(dataset.loc[selected_idx]["prediction_conf"]))

    def _get_data_based_on_task(self, task: str):
        dataset = pd.read_csv(os.path.join(self.CONFIG['data_path'], task, f"val_inference_results.csv"))
        dataset["prediction_conf"] = dataset["prediction_conf"].apply(lambda x:list(map(float, x.strip("[]").split())))
        return dataset
    
    def reset(self):
        self.seen_info = []
        self.seen_human = []
        self.seen_damage = []
        self.seen_satellite = []
        self.seen_drone = []
        
        self.isTreeCorrectlyAnswered = [False] * len(self.tasks)
        self.number_of_times_additional_data_requested = 0
        self.currentStep = 0
        
        self.task_index = 0
        self.tree_score = 0
        self.ground_truth, self.current_task_info = self.get_task_data()

    def get_task_data(self):
        task = self.tasks[self.task_index]
        return self._random_select_record_from_dataset(task)
    
    def step(self):
        done = False
        task = self.tasks[self.task_index]
        if self.ground_truth == self.current_task_info:
            self.isTreeCorrectlyAnswered[self.task_index] = True
            self.task_index += 1
        else:
            self.isTreeCorrectlyAnswered[self.task_index] = False
            self.task_index += 1
        
        if self.task_index >= len(self.tasks):
            if np.mean(self.isTreeCorrectlyAnswered) > 0.9:
                self.correct_answered_tree_counter +=1
            else:
                self.wrongly_answered_tree_counter +=1
            self.tree_counter += 1
            done = True
            self.task_index = 0
            task = self.tasks[self.task_index]
        else:
            task = self.tasks[self.task_index]
        
        if done:
            self.currentEpisode += 1
            info = {
                "episode_ended":True,
                "isTreeCorrectlyAnswered":np.mean(self.isTreeCorrectlyAnswered),
                "currentEpisode":self.currentEpisode,
                "tree_id":self.tree_counter,
                "number_of_correctly_answered_trees":self.correct_answered_tree_counter,
                "number_of_wrongly_answered_trees":self.wrongly_answered_tree_counter
            }
        else:
            info = {
                "episode_ended":False,
                "isTreeCorrectlyAnswered":np.mean(self.isTreeCorrectlyAnswered),
                "currentEpisode":self.currentEpisode,
                "tree_id":self.tree_counter,
                "number_of_correctly_answered_trees":self.correct_answered_tree_counter,
                "number_of_wrongly_answered_trees":self.wrongly_answered_tree_counter
            }
            
        return done,info 