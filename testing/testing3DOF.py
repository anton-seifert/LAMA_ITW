import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from inputimeout import inputimeout, TimeoutOccurred
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






def test(start_configs: np.array, general_config: dict, timestamp: Optional[str] = None, anchor : Optional[bool] = False):
    #reading from gereral_cofig
    env = general_config.get("robot_model_path")
    algo = general_config.get("algo")

    if timestamp is None:
        timestamp = general_config.get("timestamp")

    trained_model_path = f"models/{algo}_training/best_models/best_model_{timestamp}/best_model.zip"

    #deciding render mode wether its rewatch of failed episodes
    if anchor == True:
        render_mode = "human"
    else:
        render_mode = None

    Env_Class = ENV_MAP[env]
    Algo_Class = ALGO_MAP[algo]
    

    #Map Env
    #env erstelen

    #vecenv erstellen
    env = make_vec_env(
    Env_Class, 
    n_envs=1, 
    env_kwargs={"config": general_config, "render_mode": render_mode} # no render
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

    for i, config in enumerate(start_configs):
        print(f"i: {i} config: {config}")
        final_info = None

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
        
        final_info = info
        print(f"{final_info.get("stayed_in_target")}  steps: {final_info.get("steps_passed")}  steps_in_goal_space: {final_info.get("total_steps_passed_in_goal_range")}")
        # Daten sammeln
        rewards.append(total_reward)
        # ".get()" mit Default-Wert verhindert Crashs, falls Key fehlt
        successes.append(final_info.get("stayed_in_target"))
        distances.append(final_info.get("total_distance"))
        duration.append(final_info.get("total_steps_passed_in_goal_range", 0))
        steps.append(final_info.get("steps_passed"))

    #cancel after rewatching failed runs (breaking out of recursion)
    
    
    successes = np.array(successes)
    print("Fehlgeschlagene Start Configs")
    failed_runs = (start_configs[~successes])
    #print(failed_runs) 
    print()
    mean_reward = np.mean(rewards)
    success_rate = np.mean(successes)
    mean_steps = np.mean(steps)
    mean_duration_in_target = np.mean(duration)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    #printing results
    print(f"succes_rate: {success_rate*100}%")
    print(f"mean steps: {mean_steps}")

    if anchor == True:
        return
    
    try:
        # Wartet 30 Sekunden auf Input
        answer = inputimeout(prompt='Willst du den Fail nochmal sehen? (y/n): ', timeout=30)
    except TimeoutOccurred:
        # Das passiert nach 30 Sekunden ohne Eingabe
        print("\nZeit abgelaufen! Mache automatisch weiter (Nein).")
        answer = 'n'

    if answer == "yes" or answer ==  "y":
        print("rendering failed episodes")
        test(start_configs= failed_runs, general_config= general_config, timestamp= timestamp, anchor= True)
    else:
        print("closing now...")
    
    return mean_reward, success_rate, mean_duration_in_target, mean_distance, std_distance


if __name__ == "__main__":
    from configurations.config_3DOF import Settings as general_config
    from testing.test_set_3DOF import test_start_config
    test(start_configs= test_start_config, timestamp="20260214-100213", general_config = general_config)