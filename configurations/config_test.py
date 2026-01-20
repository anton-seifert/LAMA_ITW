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
    "robot_model_path": "assets/test_robot.xml",
    "render_mode": "human",
    "seed" : 42,
    "device" : "cpu",

    #ENV INFO
    "goal_distance" : 0.1,
    "max_steps": 3_000,

    "distance_reward" : 20,
    "energy_reward" : 0.2,
    "goal_reward" : 50,
    

    #TRAIN INFO
    "env" : "RobotWorldEnv",    #Options: RobotWorldEnv
    "algo" : "PPO",             #Options: PPO, DDPG, SAC, TD3
    "total_timesteps" : 1_200_000, #timesteps of WHOLE training
    #"n_train_envs" : 11,       #normally is adjusted to "PC" variable
    
    "model_kwargs": {
    "learning_rate": 3e-4,     # Standard(3e-4)often is good enough
    "n_steps": 2048,           # after n_steps the policy is updated
    #"batch_size": 64,          
    #"n_epochs": 10,
    #"gamma": 0.99,
    #"gae_lambda": 0.95,
    #"clip_range": 0.2,
    #"ent_coef": 0.0,           # Erhöhen auf 0.001, falls er zu schnell konvergiert
    
    # Architektur anpassen
    "policy_kwargs": {
        "net_arch": [256, 256], # Bigget NN for complex physics
        #"activation_fn": "torch.nn.Tanh" # Tanh ist oft besser für Continuous Control als ReLU
            }
    },

    #EVAL
    "eval_freq" : 5000,
    "n_check_envs": 5,
    "n_eval_episodes" : 25, #episodes passed in eval, should be multiple of check_envs

    #SCORE
    "score": None,

    #Always the same
    "timestamp": time.strftime("%Y%m%d-%H%M%S")

    
}