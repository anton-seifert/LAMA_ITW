from typing import Optional

#import subscripts

from envs.test_env import RobotWorldEnv

from train import test_train
from train import render

def main(config: Optional[dict] = None):

    """
    run a training cycle with specified parameters
    The new model gets evaluatet on test set
    Result are saved to ...
    """

    #TODO: load from config file
    #TODO: evaluate on train set and safe succes rate 
    #TODO: save time also on file

    #TODO: implement random gen
    random_seed = False
    

    #LOADING CONFIG
    if config != None:
        print("loading config")
        #TODO: read from file
        env_config = ...
        train_config = ...
    elif config == None:
        env_config = {"distance_reward" : 0.1}
        train_config = {"Policy" : "MlpPolicy"}

    #TRAINING
    #TODO: env creation auch ins training
    env = make_vec_env(RobotWorldEnv(config),n_envs=6, env_kwargs={"model_path":modelpath},vec_env_cls=SubprocVecEnv)
    test_train(env = env, config = config)
    #TODO: hier test run zu evaluation

    #RENDER Model
    render(config)



    

if __name__ == "__main__" :
    from configurations.config_test import Settings as config_dict
    main(config= config_dict)

    




