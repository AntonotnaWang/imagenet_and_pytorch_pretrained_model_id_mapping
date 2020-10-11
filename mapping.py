import numpy as np
import json
import scipy.io

def get_model_output_id_wnid_class_dict(jsonfilepath='data/modeloutput_id_wnid_class.json'):
    '''
    get the dict of model output ID, WNID and class name
    from the given json file
    format: {"Model Ouput ID": ["WNID", "Class"]}
    '''
    with open(jsonfilepath, 'r') as f:
        id_dict = json.load(f)
    
    return id_dict

def get_imagenet_id_wnid_class_dict(matfilepath='data/imagenet_id_wnid_class.mat'):
    '''
    get the dict of ImageNet ID, WNID and class name
    from the given mat file
    format: {"ImageNet ID": ["WNID", "class"]}, e.g. {..."233": ['n02106382', 'Bouvier_des_Flandres'], ...}
    '''
    meta=scipy.io.loadmat(matfilepath)['synsets']
    length=len(meta)
    id_dict={str(meta[0][0][0][0][0]):[meta[0][0][1][0], meta[0][0][2][0]]}
    for i in range(1, length):
        id_dict[str(meta[i][0][0][0][0])]=[meta[i][0][1][0], meta[i][0][2][0]]
        
    return id_dict

def map_model_id_to_imagenet_id(imagenet_id, modeloutput_id):
    '''
    return a dict mapping modeloutput id to imagenet id
    '''
    map_dict={}
    for imagenet_id_key in imagenet_id:
        for modeloutput_id_key in modeloutput_id:
            if modeloutput_id[modeloutput_id_key][0]==imagenet_id[imagenet_id_key][0]:
                map_dict[modeloutput_id_key]=imagenet_id_key
                break
    return map_dict

def map_imagenet_id_to_model_id(imagenet_id, modeloutput_id):
    '''
    return a dict mapping imagenet id to modeloutput id
    '''
    map_dict={}
    for imagenet_id_key in imagenet_id:
        for modeloutput_id_key in modeloutput_id:
            if modeloutput_id[modeloutput_id_key][0]==imagenet_id[imagenet_id_key][0]:
                map_dict[imagenet_id_key]=modeloutput_id_key
                break
    return map_dict
