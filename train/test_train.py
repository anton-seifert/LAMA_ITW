import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
import sys

#Add parent directory to sys.path to resolve cross-directory imports from sibling packages
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from envs.test_env import RobotWorldEnv


folder_name = "assets"
file_name = "test_robot.xml"
modelpath = os.path.join(folder_name, file_name)

save_dir = "models/ppo_training/"
os.makedirs(save_dir, exist_ok=True)


env = RobotWorldEnv(modelpath)

#Checks if Costum env corresponds GymAPI  
check_env(env)

model = PPO("MultiInputPolicy",env,device="cpu",tensorboard_log ="./ppo_test_robot_tensorboard/")

model.learn(total_timesteps=5_000,progress_bar=True)

model.save(f"{save_dir}/ppo_test_robot")