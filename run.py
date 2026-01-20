from typing import Optional

#import subscripts

from train.render import render

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
    #TODO: hier test run zu evaluation

    #RENDER Model
    render(config)




if __name__ == "__main__" :
    from configurations.config_test import Settings as config_dict
    from train.test_train import train
    main(config= config_dict)

    




