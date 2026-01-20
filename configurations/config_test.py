from envs.test_env import RobotWorldEnv
import time

"""
file to define als the parameters for a training run
if value is set to None, the standard value will be choosen
"""

Settings = {

    #GENERAL INFO
    
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
    "env" : RobotWorldEnv, #sowas sollte eig als string eingeben werden und dann in den anderen scripten gemapped
    "algo" : "PPO",
    "total_timesteps" : 10_000_000, #timesteps of WHOLE training

    #TODO: currently not included
    "n_step" : 10_000, #after n_steps the policy is updated
    "eval_freq" : 5000,
    "n_check_envs": ...,
    "n_train_envs" : ...,

    #SCORE
    "score": None,

    #Always the same
    "timestamp": time.strftime("%Y%m%d-%H%M%S")

    
}