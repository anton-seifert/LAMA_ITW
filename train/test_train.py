import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from typing import Optional
import os
import sys
import time



#Add parent directory to sys.path to resolve cross-directory imports from sibling packages
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from envs.test_env import RobotWorldEnv

def train(config: Optional[dict] = None):

    #TODO: variablem mit config.get() holen
    # first value is key in settings dict, secodn is fallback default value
    modelpath = config.get("robot_model_path", "assets/test_robot.xml")
    env = config.get("env", RobotWorldEnv)
    env_kwargs = ... #brauch man glaub net
    vec_env_cls = ... #was bist du?
    device = config.get("device", "cpu")
    verbose = ...#was bist du?
    total_timesteps = config.get("total_timesteps", 123_456)
    algorithm = ...
    seed = config.get("seed", 42)
    eval_freq = config.get("eval_freq", 5555)
    timestamp = config.get("timestamp", 000000000)
    algo = config.get("algo")

    #modelpath = os.path.join(folder_name, file_name)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    #save_dir = "models/ppo_training/"
    #os.makedirs(save_dir, exist_ok=True)

    env_control = RobotWorldEnv(config=config)
    #Checks if Costum env corresponds GymAPI  
    check_env(env_control)
    env_control.close()

    vec_check_env = make_vec_env(env, seed = seed, n_envs=8, env_kwargs={"config":config}, vec_env_cls=SubprocVecEnv,monitor_dir=f"./monitor_logs/{algo}_training/logs_check_{timestamp}")
   # vec_check_env = VecMonitor(vec_check_env, filename=f"./monitor_logs/logs_check{timestamp}" )
    vec_check_env = VecNormalize(vec_check_env, norm_obs=True, norm_reward=False, training=False)

    vec_train_env = make_vec_env(env, seed = seed,n_envs=8, env_kwargs={"config":config},vec_env_cls=SubprocVecEnv, monitor_dir=f"./monitor_logs/{algo}_training/logs_train_{timestamp}")
   # vec_train_env = VecMonitor(vec_train_env, filename=f"./monitor_logs/logs_train{timestamp}")    # is_sucessfull, usw. noch hinzufügen  , info_keywords=("distance") 
    vec_train_env = VecNormalize(vec_train_env , norm_obs=True, norm_reward=True)
   

    eval_callback = EvalCallback(vec_check_env, best_model_save_path=f"models/{algo}_training/best_models/best_model_{timestamp}", eval_freq=eval_freq, deterministic=True, render=False)

    #TODO: also stark algorithm with the config and change parameters of algorithm object
    model = PPO("MultiInputPolicy" ,vec_train_env, device= device, tensorboard_log=f"./tensorboard/{algo}_training/tensorboard_{timestamp}", verbose=0)

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