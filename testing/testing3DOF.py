import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO, SAC, TD3, DDPG
import os
import sys
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from envs.test_env import RobotWorldEnv as DOF2_env
from envs.DOF3_env import RobotWorldEnv as DOF3_env

ALGO_MAP = {
   "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG
}

ENV_MAP = {
    "assets/test_robot.xml" : DOF2_env,
    "assets/test_robot_3DOF.xml" : DOF3_env
}

test_start_config = np.array([
    {"start_pos": [1.7, -1.7, 1.7], "start_vel": [0.1, 0.1, 0.1], "target_pos": [0.3, 0.3, 0.3]},
    {"start_pos": [ 1.7, -1.7,  1.7], "start_vel": [0.1, 0.1 ,0.1], "target_pos": [0.5, 0.5, 0.5]},
    {"start_pos": [1.7, -1.7, 1.7],"start_vel": [0.1, 0.1, 0.1],"target_pos": [0.5, 0.5, 0.5]}
    
])






def test(start_configs: np.array, general_config: dict, timestamp: Optional[str] = None, ):
    #reading from gereral_cofig
    env = general_config.get("robot_model_path")
    algo = general_config.get("algo")

    if timestamp is None:
        timestamp = general_config.get("timestamp")

    trained_model_path = f"models/{algo}_training/best_models/best_model_{timestamp}/best_model.zip"

    


    Env_Class = ENV_MAP[env]
    Algo_Class = ALGO_MAP[algo]
    

    #Map Env
    #env erstelen

    #vecenv erstellen
    env = make_vec_env(
    Env_Class, 
    n_envs=1, 
    env_kwargs={"config": general_config} # no render
    )

    #normalize_vec_env  
    stats_path = f"models/{algo}_training/best_models/best_model_{timestamp}/vec_normalize.pkl"
    if os.path.exists(stats_path):
        print(f"Lade Normalisierungs-Stats von {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = False     # Keine Updates der Statistik mehr (frieren)
        env.norm_reward = False  # Wir wollen echte Rewards sehen, keine skalierten
    else:
        print("WARNUNG: Keine VecNormalize Stats gefunden! Roboter könnte zucken.")

    #normalisieren

    #load model
    model = Algo_Class.load(trained_model_path, env = env, device="cpu")


    #test different start config
    rewards = []
    successes = []
    distances = []
    duration = []
    steps = []

    for config in start_configs:
        env.env_method("set_reset_options", config)
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)

            if isinstance(done, np.ndarray):
                done = done[0]
                info = info[0]
                reward = reward[0]
            
            total_reward += reward

        # Daten sammeln
        rewards.append(total_reward)
        # ".get()" mit Default-Wert verhindert Crashs, falls Key fehlt
        successes.append(info.get("stayed_in_target", False))
        distances.append(info.get("total_distance"))
        duration.append(info.get("total_steps_passed_in_goal_range", 0))
        steps.append(info.get("steps_passed"))

    mean_reward = np.mean(rewards)
    success_rate = np.mean(successes)
    mean_steps = np.mean(steps)
    mean_duration_in_target = np.mean(duration)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    #printing results
    print(f"succes_rate: {success_rate}")
    print(f"mean steps: {mean_steps}")

    return mean_reward, success_rate, mean_duration_in_target, mean_distance, std_distance


if __name__ == "__main__":
    from configurations.config_3DOF import Settings as general_config
    test(start_configs= test_start_config, timestamp="20260204-162342", general_config = general_config)