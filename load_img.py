import os
from PIL import Image
import numpy as np

def get_img_names_and_labels_from_imagenet_sample(data_filepath="data/sample_1000"):
    img_files=os.listdir(data_filepath)
    img_files.sort()
    
    labels = []
    for idx, file in enumerate(img_files):
        labels.append(int(file.split("_")[0]))
    
    print("There are "+str(len(img_files))+" imgs, and "+str(len(np.unique(np.array(labels))))+" classes.")
    
    return img_files, labels

def load_img_from_imagenet_sample_by_index(index, imagenet_labels=None,
                                           data_filepath="data/sample_1000"):
    '''
    INPUT:
    index: (if data_filepath="data/sample_1000") 0~999, which image file to load
    imagenet_labels: a dict, len=1000, {"ImageNet ID": ["WNID", "class"]}, e.g. {..."233": ['n02106382', 'Bouvier_des_Flandres'], ...}
    data_filepath:
    (name format of imgs: [ImageNet_ID]_[WNID]_[N].JPEG, N: 0~4, e.g. "141_n02104029_3.JPEG" means ImageNet ID: 141, WNID: n02104029,
    3nd pic of this class, and its class is 'kuvasz'.)
    
    OUTPUT:
    the img file
    '''
    img_files, labels = get_img_names_and_labels_from_imagenet_sample(data_filepath)
    
    load_img = Image.open(data_filepath+"/"+img_files[index]).convert('RGB')
    
    if imagenet_labels is not None:
        print("load img "+data_filepath+"/"+img_files[index]+\
              "\nImageNet ID: "+str(labels[index])+\
              "\nWNID and class: "+str(imagenet_labels[str(labels[index])]))
    else:
        print("load img "+data_filepath+"/"+img_files[index]+\
              "\nImageNet ID: "+str(labels[index]))
    
    return load_img

def load_img_from_imagenet_sample_by_class(imagenet_id, imagenet_labels=None,
                                           data_filepath="data/sample_1000"):
    '''
    INPUT:
    imagenet_id: 1~1000, indicating which class to load, we'll randomly choice one img in the class
    imagenet_labels: a dict, len=1000, {"ImageNet ID": ["WNID", "class"]}, e.g. {..."233": ['n02106382', 'Bouvier_des_Flandres'], ...}
    data_filepath:
    (name format of imgs: [ImageNet_ID]_[WNID]_[N].JPEG, N: 0~4, e.g. "141_n02104029_3.JPEG" means ImageNet ID: 141, WNID: n02104029,
    3nd pic of this class, and its class is 'kuvasz'.)
    
    OUTPUT:
    the img file
    '''
    img_files, labels = get_img_names_and_labels_from_imagenet_sample(data_filepath)
    
    idxs=np.where(np.array(labels)==imagenet_id)[0]
    chosen_idx=int(np.random.choice(idxs, 1))
    
    load_img = Image.open(data_filepath+"/"+img_files[chosen_idx]).convert('RGB')
    
    if imagenet_labels is not None:
        print("load img "+data_filepath+"/"+img_files[chosen_idx]+\
              "\nImageNet ID: "+str(labels[chosen_idx])+\
              "\nWNID and class: "+str(imagenet_labels[str(labels[chosen_idx])]))
    else:
        print("load img "+data_filepath+"/"+img_files[chosen_idx]+\
              "\nImageNet ID: "+str(labels[chosen_idx]))
    
    return load_img