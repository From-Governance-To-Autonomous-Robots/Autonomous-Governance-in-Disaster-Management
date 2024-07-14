import envs.taskEnvTrain
import envs.taskEnvVal
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import time
import wandb
import numpy as np 
import gym 
import os 
import logging
import pandas as pd 
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from utils.utils import load_config,make_env
from utils.wandb_context import prefixed_wandb_log
from callbacks.wandbCallback import MetricsCallback
import argparse
import pdb

def train_and_evaluate(CONFIG):
    num_envs = CONFIG["environment"]["train"]["num_parallel_env"]
    env_ids = [CONFIG["environment"]["train"]["name"]] * num_envs
    envs = [make_env(env_id) for env_id in env_ids]
    env = SubprocVecEnv(envs)
    
    env.reset()

    # A2C specific hyperparams from config
    a2c_hyperparams = CONFIG['A2C_hyperparameters']

    model = A2C(CONFIG['policy'], env, verbose=1,
        learning_rate = a2c_hyperparams['learning_rate'],
        gamma = a2c_hyperparams['gamma'],
        n_steps = a2c_hyperparams['n_steps'],
        ent_coef = a2c_hyperparams['ent_coef'],
        vf_coef = a2c_hyperparams['vf_coef'],
        max_grad_norm = a2c_hyperparams['max_grad_norm'],
        policy_kwargs = {"optimizer_kwargs":{"eps":1e-7}}
        ) 

    #model = A2C(CONFIG['policy'], env, verbose=1) 

    total_timesteps = CONFIG['total_timesteps'] 
    model.set_env(env=env)
    callbacks = MetricsCallback()
    
    model.learn(total_timesteps=total_timesteps,reset_num_timesteps=False,callback=callbacks)
    wandb.finish()

def main(CONFIG):
    train_and_evaluate(CONFIG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config',type=str,default="/home/julian/git-repo/juliangdz/GovernanceIRP/Autonomous-Governance-in-Disaster-Management/rl_decision_maker/configs/config.yaml")
    args = parser.parse_args()
    CONFIG = load_config(args.config)

    log_dir = os.path.join(os.getcwd(),CONFIG['log_dir'])
    os.makedirs(log_dir,exist_ok=True)
    
    wandb.init(
        config=CONFIG,
        entity=CONFIG['wandb']['entity'],
        project=CONFIG['wandb']['project'],
        name=CONFIG['wandb']['name'],
        monitor_gym=True,       # automatically upload gym environements' videos
        save_code=True,
    )
    
    main(CONFIG)