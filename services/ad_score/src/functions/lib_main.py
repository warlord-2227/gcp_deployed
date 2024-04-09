import utilities

def pre_process_build_model(config,dataset):
    accuracy, cm = utilities.build_model(dataset,config)
    
    return {'accuracy':accuracy, 'confusion matrix':cm}