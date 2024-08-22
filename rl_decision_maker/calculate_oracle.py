import numpy as np
import yaml
import pandas as pd 
from argparse import ArgumentParser
import os 
import random
import pdb
import wandb
from utils.utils import load_config
from utils.log_helper import log_aggregate_stats

class OracleSequenceValEnv():
    def __init__(self):
        super(OracleSequenceValEnv, self).__init__()
        self.CONFIG = load_config("/home/julian/git-repo/juliangdz/GovernanceIRP/Autonomous-Governance-in-Disaster-Management/rl_decision_maker/configs/oracle_config.yaml")
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
        self.isTreeAdditionalDataRequested = [False] * len(self.tasks)
        self.isTreeWronglyAnswered = [False] * len(self.tasks)
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
        dataset = pd.read_csv(os.path.join(self.CONFIG['data_path'], task, f"train_inference_results.csv"))
        dataset["prediction_conf"] = dataset["prediction_conf"].apply(lambda x: list(map(float, x.strip("[]").split())))
        return dataset
    
    def reset(self):
        self.seen_info = []
        self.seen_human = []
        self.seen_damage = []
        self.seen_satellite = []
        self.seen_drone = []
        
        self.isTreeCorrectlyAnswered = [False] * len(self.tasks)
        self.isTreeAdditionalDataRequested = [False] * len(self.tasks)
        self.isTreeWronglyAnswered = [False] * len(self.tasks)
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
            self.isTreeWronglyAnswered[self.task_index] = False
            self.tree_score += 1
            self.task_index += 1
        else:
            self.isTreeCorrectlyAnswered[self.task_index] = False
            self.isTreeWronglyAnswered[self.task_index] = True
            self.tree_score -=5
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
                "episode_ended": True,
                "isTreeCorrectlyAnswered": np.mean(self.isTreeCorrectlyAnswered),
                "isTreeWronglyAnswered":np.mean(self.isTreeWronglyAnswered),
                "currentEpisode": self.currentEpisode,
                "tree_id": self.tree_counter,
                "tree_score":self.tree_score,
                "number_of_correctly_answered_trees": self.correct_answered_tree_counter,
                "number_of_wrongly_answered_trees": self.wrongly_answered_tree_counter
            }
        else:
            info = {
                "episode_ended": False,
                "isTreeCorrectlyAnswered": np.mean(self.isTreeCorrectlyAnswered),
                "isTreeWronglyAnswered":np.mean(self.isTreeWronglyAnswered),
                "currentEpisode": self.currentEpisode,
                "tree_id": self.tree_counter,
                "tree_score":self.tree_score,
                "number_of_correctly_answered_trees": self.correct_answered_tree_counter,
                "number_of_wrongly_answered_trees": self.wrongly_answered_tree_counter
            }
            
        return done, info


def main():
    parser = ArgumentParser()
    parser.add_argument('-config',type=str,default="/home/julian/git-repo/juliangdz/GovernanceIRP/Autonomous-Governance-in-Disaster-Management/rl_decision_maker/configs/oracle_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Initialize wandb
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], name=config['wandb']['name'])
    
    env = OracleSequenceValEnv()
    total_steps = config['total_timesteps']  # or 1000000
    log_interval = total_steps // config['log_interval']
    step_count = 0
    episode_count = 0
    
    collected_dictionary = {
        "episode_ended":[],
        "isTreeCorrectlyAnswered":[],
        "isTreeWronglyAnswered":[],
        "currentEpisode":[],
        "tree_id":[],
        "tree_score":[],
        "number_of_correctly_answered_trees":[],
        "number_of_wrongly_answered_trees":[]
    }
    number_of_correctly_answered_trees=0
    number_of_wrongly_answered_trees=0
    tree_id=0
    while step_count < total_steps:
        done, info = env.step()
        step_count += 1

        if done:
            episode_count += 1
            collected_dictionary["isTreeCorrectlyAnswered"].append(info["isTreeCorrectlyAnswered"])
            collected_dictionary["isTreeWronglyAnswered"].append(info["isTreeWronglyAnswered"])
            collected_dictionary["currentEpisode"].append(info["currentEpisode"])
            collected_dictionary["tree_id"].append(info["tree_id"])
            collected_dictionary['tree_score'].append(info["tree_score"])
            collected_dictionary["number_of_correctly_answered_trees"].append(info["number_of_correctly_answered_trees"])
            collected_dictionary["number_of_wrongly_answered_trees"].append(info["number_of_wrongly_answered_trees"])
#            wandb.log(info)
            env.reset()
        if (step_count) % log_interval == 0 or step_count == total_steps -1 :
            log_aggregate_stats(collected_dictionary,key="isTreeCorrectlyAnswered",log_string="isTreeCorrectlyAnswered",step=step_count)
            log_aggregate_stats(collected_dictionary,key="isTreeWronglyAnswered",log_string="isTreeWronglyAnswered",step=step_count)
            log_aggregate_stats(collected_dictionary,key="tree_score",log_string="tree_score",step=step_count)
            number_of_correctly_answered_trees=max(collected_dictionary["number_of_correctly_answered_trees"])
            number_of_wrongly_answered_trees=max(collected_dictionary["number_of_wrongly_answered_trees"])
            tree_id =max(collected_dictionary["tree_id"])
            collected_dictionary = {
                "episode_ended":[],
                "isTreeCorrectlyAnswered":[],
                "isTreeWronglyAnswered":[],
                "currentEpisode":[],
                "tree_id":[],
                "tree_score":[],
                "number_of_correctly_answered_trees":[],
                "number_of_wrongly_answered_trees":[]
            }
    wandb.log({"number_of_correctly_answered_trees":number_of_correctly_answered_trees})
    wandb.log({"number_of_wrongly_answered_trees":number_of_wrongly_answered_trees})
    wandb.log({"completed_trees":tree_id})
    wandb.finish()

if __name__ == "__main__":
    main()
