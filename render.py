import gymnasium as gym
from stable_baselines3 import PPO, TD3, DDPG, SAC 
import os
import sys
import time
from typing import Optional
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecNormalize
from utils import plot_live_dict



#Add parent directory to sys.path to resolve cross-directory imports from sibling packages
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


def render(config: Optional[dict] = None, 
           robot_model_path: Optional[str] = None, 
           trained_model_path: Optional[str] = None, 
           render_mode: Optional[str] = None,  # 'human' oder 'video'
           timestamp: Optional[str] = None):

    # --- 1. Konfiguration laden ---
    environment = config.get("robot_model_path")
    algo = config.get("algo")
    
    # Priority: Funktions-Argument > Config-Datei > Default
    if timestamp is None:
        timestamp = config.get("timestamp")

    if render_mode is None:
        render_mode = config.get("render_mode", "human")

    if trained_model_path is None:
        trained_model_path = f"models/{algo}_training/best_models/best_model_{timestamp}/best_model.zip"
    
    
    
    print(f"running {trained_model_path}")
    print(f"TIMESTAMP: {timestamp}")
    Env_Class = ENV_MAP[environment]
    Algo_Class = ALGO_MAP[algo]

    
    # Standard-Werte für "Human"
    env_render_mode = "human"  # Das, was wir an die Klasse übergeben
    use_video_wrapper = False
    sleep_time = 0.05        # Damit man im Viewer was erkennt (ca. 60-100 FPS)

    # Anpassung für "Video"
    if render_mode == "video":
        print("Modus: VIDEO-AUFNAHME (Headless)")
        env_render_mode = "rgb_array" # Wichtig für den Wrapper!
        use_video_wrapper = True
        sleep_time = 0                # Video soll so schnell wie möglich rendern
    else:
        print("Modus: LIVE VIEWER")

    # --- 3. Environment erstellen ---
    # make vec env with rifght env_render_mode
    env = make_vec_env(
    Env_Class, 
    n_envs=1, 
    env_kwargs={"config": config, "render_mode": env_render_mode} # render_mode ist für Video wichtig
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

    # Falls Video gewünscht -> Wrapper drumwickeln
    if use_video_wrapper:
        video_folder = f"videos/{algo}_{timestamp}"
        
        # VecVideoRecorder funktioniert anders als der Gym RecordVideo!
        # Er braucht die Länge in Steps, nicht Episoden.
        env = VecVideoRecorder(
            env,
            video_folder=video_folder,
            record_video_trigger=lambda step: step == 0, # Nimmt ab dem allerersten Step auf
            video_length=2000, # Maximale Länge des Videos (in Steps)
            name_prefix="render_video"
    )
    # --- 4. Modell laden & Loop ---
    #model_path must bei vecnorm
    model = Algo_Class.load(trained_model_path, env = env, device="cpu")
    
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        
        #data_dict = env.get_rewards()

        #plot_live_dict(data_dict,timestamp=timestamp)

        if done[0]:
            #info[0] weil env immer ne list an envs wieder gibt
            print(f"tcp: {info[0]['tcp']}")           
            print(f"target: {info[0]['target']}")
            print(f"distance: {info[0]['distance']}")
            print(f"energy: {info[0]['energy']}")
            print(f"steps passed: {info[0]['steps_passed']}")
            print(f"truncated because of no distance improvement: {info[0]['truncated_distance']}")
            print(f"total_reward: {reward}")
            print(f"steps passed in goal space: {info[0]['total_steps_passed_in_goal_range']}")
            print(f"floor crash: {info[0]['floor_crash']}")
            print(f"terminated: {info[0]['stayed_in_target']}")



        # --- 5. Dynamisches Warten ---
        if sleep_time > 0:
            time.sleep(sleep_time)

    # WICHTIG: Environment schließen (speichert das Video final ab)
    time.sleep(10)
    env.close()
    
    if use_video_wrapper:
        print(f"Video gespeichert in: {os.path.abspath(video_folder)}")

if __name__ == "__main__":
    #hier einstellen aus welcher config geladen werden solll, muss zum roboter passen #configurations.config_test oder configurations.config_3DOF
    from configurations.config_3DOF import Settings as config_dict
    #render_mode human für mujoco viever, "video" for video
    render(config= config_dict, timestamp="20260206-105812", render_mode= "human")