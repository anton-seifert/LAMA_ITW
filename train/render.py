import gymnasium as gym
from stable_baselines3 import PPO # change to right policy
import os
import sys
import time
#Add parent directory to sys.path to resolve cross-directory imports from sibling packages
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from envs.test_env import RobotWorldEnv
def render(robot_model_path: str, trained_model_path: str):
    # 1. Environment mit 'human' Modus erstellen
    env = RobotWorldEnv(model_path=robot_model_path, render_mode="human")

    # 2. Modell laden
    model = PPO.load(trained_model_path)

    # 3. Der Loop
    obs, _ = env.reset()
    done = False

    while not done:
        # Das Modell fragen, was zu tun ist (deterministic=True macht es stabiler)
        action, _ = model.predict(obs, deterministic=True)
        
        # Schritt ausführen (Rendering passiert automatisch in deinem step(), 
        # wenn du meinen Code von vorhin genutzt hast)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Beenden wenn fertig
        done = terminated or truncated
        
        # Optional: Ein bisschen warten, falls es zu schnell geht
        # time.sleep(0.01)
    time.sleep(10)
    env.close()

if __name__ == "__main__":
    render(robot_model_path="assets/test_robot.xml", trained_model_path=".models/ppo_training/best_models/best_model_20260118-233559/best_model.zip")