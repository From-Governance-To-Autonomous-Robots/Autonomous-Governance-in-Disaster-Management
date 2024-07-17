import wandb
from stable_baselines3.common.callbacks import BaseCallback
from utils.log_helper import log_aggregate_stats,log_to_csv,log_results_table_to_wandb
import os
from utils.wandb_context import prefixed_wandb_log
from utils.evaluate import evaluate
from utils.utils import load_config
import pdb
import numpy as np

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_counts = None
        self.current_episode_rewards = None
        self.current_episode_lengths = None
        self.collected_dictionary = {
            "episode_ended":[],
            "tree_score":[],
            "isTreeCorrectlyAnswered":[],
            "currentEpisode":[],
            "currentStep":[],
            "tree_id":[],
            "currentStepReward":[],
            "number_of_correctly_answered_trees":[],
            "number_of_wrongly_answered_trees":[],
            "number_of_times_additional_data_requested":[]
        }
        self.CONFIG = load_config()
        self.total_timesteps = self.CONFIG['total_timesteps']
        self.log_interval = self.total_timesteps // self.CONFIG['log_interval']
        self.eval_interval = self.total_timesteps // self.CONFIG['eval_interval']
        self.train_csv_file_path = os.path.join(os.getcwd(),self.CONFIG["log_dir"],'train_results_summary.csv')
        self.eval_csv_file_path = os.path.join(os.getcwd(),self.CONFIG["log_dir"],'eval_results_summary.csv')
        self.m_rch_count = 0

    def _on_training_start(self):
        self.num_envs = self.model.env.num_envs
        self.episode_counts = [0] * self.num_envs
        self.current_episode_rewards = [0.0] * self.num_envs
        self.current_episode_lengths = [0] * self.num_envs
    
    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        infos = self.locals['infos']
        step = self.model.num_timesteps
        callback_return = True

        for i, info in enumerate(infos):
            if info["episode_ended"]:
                self.collected_dictionary["tree_score"].append(info["tree_score"])
                self.collected_dictionary["isTreeCorrectlyAnswered"].append(info["isTreeCorrectlyAnswered"])
                self.collected_dictionary["currentEpisode"].append(info["currentEpisode"])
                self.collected_dictionary["currentStep"].append(info["currentStep"])
                self.collected_dictionary["tree_id"].append(info["tree_id"])
                self.collected_dictionary["currentStepReward"].append(info["currentStepReward"])
                self.collected_dictionary["number_of_correctly_answered_trees"].append(info["number_of_correctly_answered_trees"])
                self.collected_dictionary["number_of_wrongly_answered_trees"].append(info["number_of_wrongly_answered_trees"])
                self.collected_dictionary["number_of_times_additional_data_requested"].append(info["number_of_times_additional_data_requested"])
            
            if info["isTreeCorrectlyAnswered"] > 0.9:
                cumulative_data = [
                    i,
                    info["tree_score"],
                    info["isTreeCorrectlyAnswered"],
                    info["currentEpisode"],
                    info["currentStep"],
                    info["tree_id"],
                    info["currentStepReward"],
                    info["number_of_correctly_answered_trees"],
                    info["number_of_wrongly_answered_trees"],
                    info["number_of_times_additional_data_requested"]
                ]
                log_to_csv(cumulative_data,self.train_csv_file_path)

        # for i in range(len(dones)):
        #     if np.sum(dones[i]) == self.num_envs: 
        #         self.model.env.reset()

        with prefixed_wandb_log("Train"):
            if (step) % self.log_interval == 0 or step == self.total_timesteps - 1:
                log_aggregate_stats(self.collected_dictionary,key="tree_score",log_string="tree_score",step=step)
                log_aggregate_stats(self.collected_dictionary,key="isTreeCorrectlyAnswered",log_string="isTreeCorrectlyAnswered",step=step)
                log_aggregate_stats(self.collected_dictionary,key="currentEpisode",log_string="currentEpisode",step=step)
                log_aggregate_stats(self.collected_dictionary,key="tree_id",log_string="tree_id",step=step)
                log_aggregate_stats(self.collected_dictionary,key="currentStepReward",log_string="currentStepReward",step=step)
                log_aggregate_stats(self.collected_dictionary,key="number_of_correctly_answered_trees",log_string="number_of_correctly_answered_trees",step=step)
                log_aggregate_stats(self.collected_dictionary,key="number_of_wrongly_answered_trees",log_string="number_of_wrongly_answered_trees",step=step)
                log_aggregate_stats(self.collected_dictionary,key="number_of_times_additional_data_requested",log_string="number_of_times_additional_data_requested",step=step)
                self.collected_dictionary = {
                    "episode_ended":[],
                    "tree_score":[],
                    "isTreeCorrectlyAnswered":[],
                    "currentEpisode":[],
                    "currentStep":[],
                    "tree_id":[],
                    "currentStepReward":[],
                    "number_of_correctly_answered_trees":[],
                    "number_of_wrongly_answered_trees":[],
                    "number_of_times_additional_data_requested":[]
                }
        
        with prefixed_wandb_log("Eval"):
            if (step) % self.eval_interval == 0 or step == self.total_timesteps - 1:
                m_count = evaluate(self.model,self.CONFIG,step,self.eval_csv_file_path)
                self.model.save(os.path.join(os.getcwd(),self.CONFIG["log_dir"],f"final_model_{step}.zip"))
                self.m_rch_count += m_count

        if self.m_rch_count == 10:
            callback_return = False
        
        # explained_variance = self.locals['self'].logger.name_to_value['train/explained_variance']
        # if explained_variance > 0.5:
        #     print('Explained Variance is Greater than Threshold. Lets Early Stop')
        #     callback_return = False
        
        return callback_return

    def _on_training_end(self):
        if os.path.exists(self.train_csv_file_path):
            log_results_table_to_wandb(self.train_csv_file_path,prefix='Train')
        if os.path.exists(self.eval_csv_file_path):
            log_results_table_to_wandb(self.eval_csv_file_path,prefix='Eval')