import gymnasium as gym
import mujoco
from stable_baselines3 import PPO, SAC, TD3, DDPG
from envs.test_env import RobotWorldEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Optional, Type

import os
import sys
import time



#Add parent directory to sys.path to resolve cross-directory imports from sibling packages
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG
}

ENV_MAP = {
    "RobotWorldEnv" : RobotWorldEnv
}

#Maps the number of train_envs to PC
N_ENV_MAP = {
    "An" : 11,  
    "Ar": 15,
    "IT" :11
}


def train(config: Optional[dict] = None):

    #variablen mit config.get() holen
    # first value is key in settings dict, secodn is fallback default value
    env = config.get("env", "RobotWorldEnv")
    vec_env_cls = ... #was bist du?
    device = config.get("device", "cpu")
    verbose = ...#was bist du?
    timesteps_per_env = config.get("timesteps_per_env", 100)
    seed = config.get("seed", 42)
    eval_freq = config.get("eval_freq", 5555)
    timestamp = config.get("timestamp", 000000000)
    algo = config.get("algo")
    n_check_envs = config.get("n_check_envs", 5)
    hyperparams = config.get("model_kwargs")
    n_eval_episodes = config.get("n_eval_episodes", 20)
    pc = config.get("PC", "An")

    
    #Mapping Variables from String
    Algo_Class : Type[BaseAlgorithm] = ALGO_MAP[algo]
    Env_Class = ENV_MAP[env]
    n_train_envs = N_ENV_MAP[pc]
    print(f"\nRunning {algo} in the {env} Environmenet on {pc} PC")

    total_timesteps = timesteps_per_env*n_train_envs
    #modelpath = os.path.join(folder_name, file_name)
    #timestamp = time.strftime("%Y%m%d-%H%M%S")

    #save_dir = "models/ppo_training/"
    #os.makedirs(save_dir, exist_ok=True)
    print("\ncreating CONTROL env:")
    env_control = Env_Class(config=config)
    #Checks if Costum env corresponds GymAPI  
    check_env(env_control)
    env_control.close()

    print(f"\ncreating {n_check_envs} CHECK envs:")
    vec_check_env = make_vec_env(Env_Class, seed = 1111, n_envs = n_check_envs, env_kwargs={"config":config}, vec_env_cls=SubprocVecEnv,monitor_dir=f"./monitor_logs/{algo}_training/logs_check_{timestamp}")
    #TODO: ist das richtig so, dass norm_reward false ist,weil im train env ist das nicht so
    vec_check_env = VecNormalize(vec_check_env, norm_obs=True, norm_reward=False, training=False)

    print(f"\ncreating {n_train_envs} TRAIN envs:")
    vec_train_env = make_vec_env(Env_Class, seed = seed,n_envs = n_train_envs, env_kwargs={"config":config},vec_env_cls=SubprocVecEnv, monitor_dir=f"./monitor_logs/{algo}_training/logs_train_{timestamp}")
    #TODO: add additional infos
    #vec_train_env = VecMonitor(vec_train_env, filename=f"./monitor_logs/logs_train{timestamp}")    # is_sucessfull, usw. noch hinzufügen  , info_keywords=("distance") 
    vec_train_env = VecNormalize(vec_train_env , norm_obs=True, norm_reward=True)
   

    eval_callback = EvalCallback(vec_check_env, best_model_save_path=f"models/{algo}_training/best_models/best_model_{timestamp}", eval_freq=eval_freq, n_eval_episodes = n_eval_episodes, deterministic=True, render=False)
    #Creating Model
    model = Algo_Class("MultiInputPolicy" ,**hyperparams, env = vec_train_env, device= device, tensorboard_log=f"./tensorboard/{algo}_training/tensorboard_{timestamp}", verbose=0)

    model.learn(total_timesteps= total_timesteps, callback=eval_callback, progress_bar=True)

    #TODO: brauchen wir das noch?
    model_save_path = f"models/{algo}_training/models/model_{timestamp}"
    #TODO: der saved die stats nicht
    stats_save_path = f"models/{algo}_training/models_vecnorm/vecnorm_{timestamp}_vecnorm.pkl"

    model.save(model_save_path)
    vec_train_env.save(stats_save_path)

    vec_train_env.close()
    vec_check_env.close()

if __name__ == '__main__':

    
   train(RobotWorldEnv)