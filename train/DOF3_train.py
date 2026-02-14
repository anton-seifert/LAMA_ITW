from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Optional, Type
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from utils import AutoSaveVecNormalize


import os
import sys
import time



#Add parent directory to sys.path to resolve cross-directory imports from sibling packages
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from envs.DOF3_env import RobotWorldEnv


ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG
}

ENV_MAP = {
    "RobotWorldEnv" : RobotWorldEnv
}

#Maps the number of train_envs to PC
N_ENV_MAP = {
    "An" : 20,  
    "Ar": 30,
    "IT" :11
}


def train(config: Optional[dict] = None):

    #variablen mit config.get() holen
    # first value is key in settings dict, secodn is fallback default value
    env = config.get("env", "RobotWorldEnv")
    device = config.get("device", "cpu")
    verbose = ...#was bist du?
    timesteps_per_env = config.get("timesteps_per_env", 100)
    seed = config.get("seed", 42)
    eval_freq = config.get("eval_freq", 5555)
    timestamp = config.get("timestamp", 000000000)
    algo = config.get("algo")
    n_check_envs = config.get("n_check_envs", 5)
    n_eval_episodes = config.get("n_eval_episodes", 20)
    pc = config.get("PC", "An")

    if algo == "PPO":
        hyperparams = config.get("model_kwargs_PPO")
    elif algo == "SAC":
        hyperparams = config.get("model_kwargs_SAC")


    
    #Mapping Variables from String
    Algo_Class : Type[BaseAlgorithm] = ALGO_MAP[algo]
    Env_Class = ENV_MAP[env]
    n_train_envs = N_ENV_MAP[pc]
    print(f"\nRunning {algo} in the {env} Environmenet on {pc} PC")

    total_timesteps = timesteps_per_env*n_train_envs

    #Init Weights and Biases
    run = wandb.init(
        project="Bachelor_Robot_RL", 
        config=config, 
        sync_tensorboard=True,
        monitor_gym= False #true if you want to upload every video done in training 
    )
    wandb.save("configurations/config_3DOF.py", base_path="configurations", policy="now")

    print("\ncreating CONTROL env:")
    env_control = Env_Class(config=config)
    #Checks if Costum env corresponds GymAPI  
    check_env(env_control)
    env_control.close()
    print("CONTROL env closed")

    print(f"\ncreating {n_check_envs} CHECK envs:")
    vec_check_env = make_vec_env(
        Env_Class,
        seed = 1111,
        n_envs = n_check_envs,
        env_kwargs={"config":config},
        vec_env_cls=DummyVecEnv,
        monitor_dir=f"./monitor_logs/{algo}_training/logs_check_{timestamp}",
        monitor_kwargs= {"info_keywords": ("distance","energy","reached_target","truncated_distance", "total_steps_passed_in_goal_range", "stayed_in_target", "floor_crash")}
        )
    #TODO: ist das richtig so, dass norm_reward false ist,weil im train env ist das nicht so
    vec_check_env = VecNormalize(vec_check_env, norm_obs=True, norm_reward=False, training=False)

    """vec_check_env = VecVideoRecorder(
        vec_check_env,
        video_folder=f"eval_videos/{timestamp}/{run_id}_eval",
        record_video_trigger=lambda x: x == 0, # Nimmt ab dem ersten Step auf
        video_length=1000,                     # Wie lang das Video max sein darf
        name_prefix=f"eval-agent"
    )"""

    print(f"\ncreating {n_train_envs} TRAIN envs:")
    
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20 , min_evals=5)

    vec_train_env = make_vec_env(
        Env_Class,
        seed = seed,
        n_envs = n_train_envs,
        env_kwargs={"config":config},
        vec_env_cls=SubprocVecEnv,
        monitor_dir=f"./monitor_logs/{algo}_training/logs_train_{timestamp}",
        monitor_kwargs= {"info_keywords": ("distance","energy","reached_target","truncated_distance")}
        )
    #TODO: add additional infos
    #vec_train_env = VecMonitor(vec_train_env, filename=f"./monitor_logs/logs_train{timestamp}")    # is_sucessfull, usw. noch hinzufügen  , info_keywords=("distance") 
    vec_train_env = VecNormalize(vec_train_env , norm_obs=True, norm_reward=True)
    
    #initialize custon callback for saving vecnorm
    vecnorm_save_path = f"models/{algo}_training/best_models/best_model_{timestamp}"
    save_stats = AutoSaveVecNormalize(vecnorm_save_path)

    eval_callback = EvalCallback(
        vec_check_env,
        best_model_save_path=f"models/{algo}_training/best_models/best_model_{timestamp}",
        eval_freq=eval_freq,
        n_eval_episodes = n_eval_episodes,
        deterministic=True,
        render=False,
        callback_after_eval=stop_train_callback,
        callback_on_new_best= save_stats
        )
    
    wandb_callback = WandbCallback(
        verbose=2,
        model_save_path=None, # <-- Dont save to cloud during training only at the end
        model_save_freq=0
    )
    #Creating Model
    model = Algo_Class("MultiInputPolicy" ,**hyperparams, env = vec_train_env, device= device, tensorboard_log=f"./tensorboard/{algo}_training/tensorboard_{timestamp}", verbose=0)

    # Normalisieren die Beobachtungen, Evaluierung benutzt nun gleiche Werte wie Training
    vec_check_env.obs_rms = vec_train_env.obs_rms

    model.learn(total_timesteps= total_timesteps, callback=[eval_callback, wandb_callback], progress_bar=True)

    #model_save_path = f"models/{algo}_training/models/model_{timestamp}"

    #stats_save_path = f"models/{algo}_training/models_vecnorm/vecnorm_{timestamp}_vecnorm.pkl"
    
    #Upload to WanB
    #TODO: safe best vecnorm
    wandb.save(f"models/{algo}_training/best_models/best_model_{timestamp}/best_model.zip", base_path=f"models/{algo}_training/best_models/best_model_{timestamp}")
    wandb.save(f"models/{algo}_training/best_models/best_model_{timestamp}/vec_normalize.pkl", base_path=f"models/{algo}_training/best_models/best_model_{timestamp}")
    
    #model.save(model_save_path)
    #vec_train_env.save(stats_save_path)

    vec_train_env.close()
    vec_check_env.close()

if __name__ == '__main__':

    
   train(RobotWorldEnv)