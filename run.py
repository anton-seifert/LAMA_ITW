from typing import Optional

#import subscripts

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
        print("loading from config")
        #TODO: read from file
        
    elif config == None:
        print("NOT loading from config, defining a new one")
        env_config = {"distance_reward" : 0.1}
        train_config = {"Policy" : "MlpPolicy"}

    #TRAINING
    
    train(config = config)
    #TODO: hier test run zu evaluation

    #RENDER Model
    #render(config)




if __name__ == "__main__" :
    from configurations.config_test import Settings as config_dict
    from train.test_train import train
    main(config= config_dict)

    




