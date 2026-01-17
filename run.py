from typing import Optional

from envs.test_env import RobotWorldEnv
from configs import config_test
from train import test_train
def main(config_path : Optional[str] = None):

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
    env_config = {}
    train_config = {}

    if config_path != None:
        print("loading config")
        #TODO: read from file
        env_config = ...
        train_config = ...
    elif config_path == None:
        env_config = {"distance_reward" : 0.1}
        train_config = {"Policy" : "MlpPolicy"}

    #TRAINING



    

if __name__ == "__main__" :
    main()

    




