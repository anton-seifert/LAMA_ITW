from typing import Optional

#import subscripts

from render import render
from testing.testing3DOF import test
#from testing.test_set_3DOF import test_start_config

def main(config: Optional[dict] = None):

    """
    run a training cycle with specified parameters
    The new model gets evaluatet on test set
    Result are saved to ...
    """

    #TODO: evaluate on train set and safe succes rate 
    #TODO: save time also on file

    

    #LOADING CONFIG
    if config != None:
        print("loading from config")
        
    elif config == None:
        print("NOT loading from config, defining a new one")
        

    #TRAINING
    
    train(config = config)

    print("TESTING")
    #test(start_configs= test_start_config, general_config= config)

    #RENDER Model
    #render(config)






if __name__ == "__main__" :
    from configurations.config_conti import Settings as config_dict
    from train.conti_train import train
    main(config= config_dict)

    




