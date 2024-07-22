import yaml 
import gym 
import numpy as np

def load_config(file_path='/home/aaimscadmin/workspace/Autonomous-Governance-in-Disaster-Management/rl_decision_maker/configs/rl_config.yaml'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def make_env(env_id):
    def _init():
        env = gym.make(env_id, render_mode="human")
        return env
    return _init
