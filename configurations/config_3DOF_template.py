from envs.test_env import RobotWorldEnv
import time

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
    "goal_distance" : 0.05,
    "max_steps": 500,

    "distance_reward" : 20,
    "energy_reward" : 1, #in gymnaisum typically one thenth of distance_reward
    "in_range_reward" : 50,
    "goal_reward" : 5000,
    "truncated_distance_reward" : 500,
    "duration_in_target" : 50, #steps need to pass in goal area for env to terminate


    #TRAIN INFO
    "env" : "RobotWorldEnv",    #Options: RobotWorldEnv
    "algo" : "SAC",             #Options: PPO, DDPG, SAC, TD3
    "timesteps_per_env" : 1_00_000, #timesteps of steps per env, should be multiples of "eval_freq"
    #"n_train_envs" : 11,       #normally is adjusted to "PC" variable
    "truncated_distance_steps": 50,

    
    "model_kwargs_PPO": {
    "learning_rate": 3e-4,     # Standard(3e-4)often is good enough
    "n_steps": 2048,           # after n_steps the policy is updated
    "batch_size": 512,          
    #"n_epochs": 10,
    #"gamma": 0.99,
    #"gae_lambda": 0.95,
    #"clip_range": 0.2,
    #"ent_coef": 0.0,          # Erhöhen auf 0.001, falls er zu schnell konvergiert
    },

    "model_kwargs_SAC": {
    "learning_rate": 0.0003,
    "buffer_size": 1000000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_entropy": "auto"
    },

    # Architektur anpassen
    "policy_kwargs": {
        "net_arch": [256, 256], # Bigget NN for complex physics
        #"activation_fn": "torch.nn.Tanh" # Tanh ist oft besser für Continuous Control als ReLU
    },


    #EVAL
    "eval_freq" : 10_000,
    "n_check_envs": 2,
    "n_eval_episodes" : 10, #episodes passed in eval, should be multiple of check_envs

    #SCORE
    "score": None,

    #Always the same
    "timestamp": time.strftime("%Y%m%d-%H%M%S")

    
}