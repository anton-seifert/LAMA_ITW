import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
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

    save_dir = "models/ppo_training/"
    os.makedirs(save_dir, exist_ok=True)

    env_control = RobotWorldEnv(model_path=modelpath)
    #Checks if Costum env corresponds GymAPI  
    check_env(env_control)
    env_control.close()

    vec_env = make_vec_env(env,n_envs=6, env_kwargs={"model_path":modelpath},vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO("MultiInputPolicy" ,vec_env, device="cpu" ,tensorboard_log ="./tensorboard/ppo_test_robot_tensorboard/", 
                policy_kwargs = dict(net_arch=[256, 256]),
                n_steps=4096,
                batch_size=256,
                ent_coef=0.001,
                learning_rate=lambda progress: progress * 0.0003)

    model.learn(total_timesteps=1_00_000,progress_bar=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_save_path = f"{save_dir}/ppo_test_robot_{timestamp}"
    stats_save_path = f"{save_dir}/ppo_test_robot_{timestamp}_vecnorm.pkl"

    model.save(model_save_path)
    vec_env.save(stats_save_path)

    vec_env.close()

if __name__ == '__main__':

    
    train(RobotWorldEnv)