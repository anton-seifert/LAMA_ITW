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

env = RobotWorldEnv(modelpath)

#Testen ob env der GymAPI entspricht 
check_env(env)