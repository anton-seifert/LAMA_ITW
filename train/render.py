import gymnasium as gym
from stable_baselines3 import PPO # change to right policy
import os
import sys
import time
from typing import Optional
#Add parent directory to sys.path to resolve cross-directory imports from sibling packages
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from envs.test_env import RobotWorldEnv

def render(config : Optional[dict] = None, robot_model_path: Optional[str] = None, trained_model_path: Optional[str] = None):
    # Read from Config file
    evnironment = config.get("env")
    algo = config.get("algo")
    timestamp = config.get("timestamp")
    
    if(trained_model_path == None):
        trained_model_path = f"models/{algo}_training/best_models/best_model_{timestamp}/best_model.zip"


    # 1. Environment mit 'human' Modus erstellen
    #TODO: Env mappen über string
    env = evnironment(config= config, render_mode="human")

    # 2. Modell laden
    model = PPO.load(trained_model_path, device= "cpu")

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
        time.sleep(0.001)

    time.sleep(10)
    env.close()

if __name__ == "__main__":
    from configurations.config_test import Settings as config_dict

    render(config= config_dict, trained_model_path="models/PPO_training/best_models/best_model_20260120-011810/best_model.zip")