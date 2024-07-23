from utils.utils import make_env
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import numpy as np 
import time 
from utils.log_helper import log_aggregate_stats,log_to_csv,log_results_table_to_wandb,calculate_aggregate_stats
import os
import pdb
import wandb

def evaluate(model,CONFIG,step,eval_csv_file_path):
    print('in Evaluate')
    n_envs = CONFIG["environment"]["val"]["num_parallel_env"]
    env_ids = [CONFIG["environment"]["val"]["name"]] * n_envs
    envs = [make_env(env_id) for env_id in env_ids]
    env = SubprocVecEnv(envs)
    
    n_eval_episodes = CONFIG["eval_episodes"]
    
    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    start_times = np.array([time.time()] * n_envs)  # Start times for each env
   
    collected_dictionary = {
        "episode_ended":[],
        "tree_score":[],
        "isTreeCorrectlyAnswered":[],
        "isTreeWronglyAnswered":[],
        "isTreeAdditionalDataRequested":[],
        "currentEpisode":[],
        "currentStep":[],
        "tree_id":[],
        "currentStepReward":[],
        "number_of_correctly_answered_trees":[],
        "number_of_wrongly_answered_trees":[],
        "number_of_times_additional_data_requested":[]
    }
    
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  
            state=states,
            episode_start=episode_starts,
            deterministic=False,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                if done:
                    
                    collected_dictionary["tree_score"].append(info["tree_score"])
                    collected_dictionary["isTreeCorrectlyAnswered"].append(info["isTreeCorrectlyAnswered"])
                    collected_dictionary["isTreeWronglyAnswered"].append(info["isTreeWronglyAnswered"])
                    collected_dictionary["isTreeAdditionalDataRequested"].append(info["isTreeAdditionalDataRequested"])
                    collected_dictionary["currentEpisode"].append(info["currentEpisode"])
                    collected_dictionary["tree_id"].append(info["tree_id"])
                    collected_dictionary["currentStepReward"].append(info["currentStepReward"])
                    collected_dictionary["number_of_correctly_answered_trees"].append(info["number_of_correctly_answered_trees"])
                    collected_dictionary["number_of_wrongly_answered_trees"].append(info["number_of_wrongly_answered_trees"])
                    collected_dictionary["number_of_times_additional_data_requested"].append(info["number_of_times_additional_data_requested"])
                    
                    #wandb.log({"Eval/episode_tree_score":info["tree_score"],"episode":episode_counts[i]})
                    
                    if info['isTreeCorrectlyAnswered'] > 0.9:
                        cumulative_data = [
                            i,
                            info["tree_score"],
                            info["isTreeCorrectlyAnswered"],
                            info["isTreeWronglyAnswered"],
                            info["isTreeAdditionalDataRequested"],
                            info["currentEpisode"],
                            info["currentStep"],
                            info["tree_id"],
                            info["currentStepReward"],
                            info["number_of_correctly_answered_trees"],
                            info["number_of_wrongly_answered_trees"],
                            info["number_of_times_additional_data_requested"]
                        ]
                        log_to_csv(cumulative_data,eval_csv_file_path)
                    
                    episode_counts[i] += 1
                    
        observations = new_observations
        
    log_aggregate_stats(collected_dictionary,key="tree_score",log_string="tree_score",step=step)
    log_aggregate_stats(collected_dictionary,key="isTreeCorrectlyAnswered",log_string="isTreeCorrectlyAnswered",step=step)
    log_aggregate_stats(collected_dictionary,key="isTreeWronglyAnswered",log_string="isTreeWronglyAnswered",step=step)
    log_aggregate_stats(collected_dictionary,key="isTreeAdditionalDataRequested",log_string="isTreeAdditionalDataRequested",step=step)
    log_aggregate_stats(collected_dictionary,key="currentEpisode",log_string="currentEpisode",step=step)
    log_aggregate_stats(collected_dictionary,key="tree_id",log_string="tree_id",step=step)
    log_aggregate_stats(collected_dictionary,key="currentStepReward",log_string="currentStepReward",step=step)
    log_aggregate_stats(collected_dictionary,key="number_of_correctly_answered_trees",log_string="number_of_correctly_answered_trees",step=step)
    log_aggregate_stats(collected_dictionary,key="number_of_wrongly_answered_trees",log_string="number_of_wrongly_answered_trees",step=step)
    log_aggregate_stats(collected_dictionary,key="number_of_times_additional_data_requested",log_string="number_of_times_additional_data_requested",step=step)
    
    mean_goal_rchd,std_reached = calculate_aggregate_stats(collected_dictionary["isTreeCorrectlyAnswered"])
    mean_tree_score,_ = calculate_aggregate_stats(collected_dictionary["tree_score"])

    if mean_goal_rchd >= 0.85 or mean_tree_score > 1:
        return 1
    # elif mean_goal_rchd == 0 and std_reached == 0:
    #     return 1
    else:
        return 0
