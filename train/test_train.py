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

def train(env, config: Optional[dict] = None):
    #TODO: variablem mit config.get() holen
    folder_name = "assets"
    file_name = "test_robot.xml"
    modelpath = os.path.join(folder_name, file_name)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    save_dir = "models/ppo_training/"
    os.makedirs(save_dir, exist_ok=True)

    env_control = RobotWorldEnv(model_path=modelpath)
    #Checks if Costum env corresponds GymAPI  
    check_env(env_control)
    env_control.close()

    vec_check_env = make_vec_env(env, n_envs=1,env_kwargs={"model_path":modelpath}, vec_env_cls=DummyVecEnv)
    vec_check_env = VecMonitor(vec_check_env, filename=f"./monitor_logs/logs_check{timestamp}" )
    vec_check_env = VecNormalize(vec_check_env, norm_obs=True, norm_reward=False, training=False)

    vec_train_env = make_vec_env(env,n_envs=1, env_kwargs={"model_path":modelpath},vec_env_cls=DummyVecEnv)
    vec_train_env = VecMonitor(vec_train_env, filename=f"./monitor_logs/logs_train{timestamp}")    # is_sucessfull, usw. noch hinzufügen  , info_keywords=("distance") 
    
    vec_train_env = VecNormalize(vec_train_env , norm_obs=True, norm_reward=True)
   

    eval_callback = EvalCallback(vec_check_env, best_model_save_path=f".models/ppo_training/best_models/best_model_{timestamp}", eval_freq=5000,deterministic=True, render=False)


    model = PPO("MultiInputPolicy" ,vec_train_env , device="cpu" ,tensorboard_log ="./tensorboard/ppo_test_robot_tensorboard/")

    model.learn(total_timesteps=1_00_000,callback=eval_callback,progress_bar=True)

    
    model_save_path = f"{save_dir}/ppo_test_robot_{timestamp}"
    stats_save_path = f"{save_dir}/ppo_test_robot_{timestamp}_vecnorm.pkl"

    model.save(model_save_path)
    vec_train_env.save(stats_save_path)

    vec_train_env.close()

if __name__ == '__main__':

    
   train(RobotWorldEnv)