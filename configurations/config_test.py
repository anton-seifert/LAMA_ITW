from envs.test_env import RobotWorldEnv

"""
file to define als the parameters for a training run
if value is set to None, the standard value will be choosen
"""

Settings = {

    #GENERAL INFO
    
    "robot_model_path": "assets/test_robot.xml",
    "trained_model_path": "lalal",
    "trained_model_name": "lalal",
    "render_mode": "human",
    "seed" : 42,
    "device" : "cpu",

    #ENV INFO
    "goal_distance" : 0.1,
    "max steps": 1_000,

    "distance_reward" : 20,
    "energy_reward" : 0.2,
    "goal_reward" : 50,
    

    #TRAIN INFO
    "env" : RobotWorldEnv,
    "total_timesteps" : 100_000, #timesteps of WHOLE training
    "n_step" : 10_000, #after n_steps the policy is updated
    "eval_freq" : 5000,

    #SCORE
    "score": None
}