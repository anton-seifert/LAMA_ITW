from envs.test_env import RobotWorldEnv
import time
import torch.nn as nn
import numpy as np
"""
file to define als the parameters for a training run
if value is set to None, the standard value will be choosen
"""
#TODO: bei None werden die standard werte gewählt
Settings = {

    #GENERAL INFO
    "PC" : "An", #Options: An, Ar, IT
    "robot_model_path": "assets/test_robot_3DOF.xml",
    "render_mode": "human",
    "seed" : 42,
    "device" : "cpu",


    #ENV INFO
    "goal_distance" : 0.01,
    "max_steps": 250,

    "distance_reward" : 73,
    "energy_reward" : 1,       #in gymnaisum typically one thenth of distance_reward
    "in_range_reward" : 58,     #reward while staying in target range
    "goal_reward" : 219,        #reward after succesfull termination
    "joint_limit_reward" : 15,   #punishment if get close to joint limits
    "singularity_reward_factor" : 0, #actually punishes velocity, velocity is infinitly high in singularities
    "truncated_distance_reward" : 285,
    "crash_reward":  46,
    "floor_distance_reward" :  26,


    #TRAIN INFO
    "env" : "RobotWorldEnv",    #Options: RobotWorldEnv
    "algo" : "PPO",             #Options: PPO, DDPG, SAC, TD3
    "timesteps_per_env" : 250_000, #timesteps of steps per env, should be multiples of "eval_freq"
    #"n_train_envs" : 11,       #normally is adjusted to "PC" variable
    "truncated_distance_steps": 75,
    "duration_in_target" : 50, #steps need to pass in goal area for env to terminate
    "target_angle_train": 1*np.pi,
    "target_angle_check": 2*np.pi,

    
    "model_kwargs_PPO": {
    "learning_rate": 3e-4,     # Standard(3e-4)often is good enough
    "n_steps": 1024,           # after n_steps the policy is updated
    "batch_size": 512,          
    #"n_epochs": 10,
    #"gamma": 0.99,
    #"gae_lambda": 0.95,
    "clip_range": 0.2,
    #"ent_coef": 0.0, 
    
    # Architektur anpassen
    "policy_kwargs": {
        "net_arch": [64, 64], # Bigget NN for complex physics
        "activation_fn": nn.Tanh, # Tanh ist oft besser für Continuous Control als ReLU
        "ortho_init": True,
            }
    },

    "model_kwargs_SAC": {
    "learning_rate": 3e-4,      # Standard: 3e-4 (oft sehr stabil)
    "buffer_size": 500_000,   # WICHTIG: SAC speichert alte Erfahrungen. Achtung RAM-Verbrauch!
    "batch_size": 256,          # Standard: 256
    "tau": 0.005,               # "Soft Update" Faktor für das Target-Netzwerk
    "gamma": 0.99,              # Discount factor
    "learning_starts": 100,     # Warte 100 Steps, bevor das Lernen beginnt (Daten sammeln)
    
    # Wie oft wird gelernt?
    "train_freq": 64,            # Update nach jedem Step (sehr rechenintensiv!)
    "gradient_steps": 64,        # 1 Gradienten-Update pro Step
    
    # Entropie (die "Neugier" des Agenten)
    # "ent_coef": "auto",       # Standard ist "auto", SAC lernt die Temperatur selbst

    # Architektur anpassen
    "policy_kwargs": {
        # SAC nutzt zwei Netze: pi (Actor) und qf (Critic/Q-Function)
        # Standard ist bei SAC größer als bei PPO: [256, 256]
        "net_arch": dict(pi=[256, 256], qf=[256, 256]), 
        
        "activation_fn": nn.ReLU, # SAC nutzt standardmäßig ReLU (PPO oft Tanh)
        
        # Optional: State Dependent Exploration (gSDE)
        # Sehr stark für MuJoCo Roboter, macht Bewegungen flüssiger
        # "use_sde": False, 
    }
    },
    
    #EVAL
    "eval_freq" : 15000,
    "n_check_envs": 2,
    "n_eval_episodes" : 20, #episodes passed in eval, should be multiple of check_envs

    #SCORE
    "score": None,

    #Always the same
    "timestamp": time.strftime("%Y%m%d-%H%M%S")
}