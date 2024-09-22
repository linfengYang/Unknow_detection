import os
import numpy as np
import logging
import random
import pandas as pd
from sklearn import model_selection
logger = logging.getLogger(__name__)

'''
    funciton
        读取数据集中的特征和数据
    return
        data,labels
'''
def load_from_csv(dirs):
    labels=[]
    data=[]
    name_dict={'Air Conditioner':0,'Compact Fluorescent Lamp':1,'Fan':2,'Fridge':3,'Hairdryer':4,'Heater':5,'Incandescent Light Bulb':6,'Laptop':7,'Microwave':8,'Vacuum':9,'Washing Machine':10}
    for folder_name in os.listdir(dirs):        #Air Conditioner,Fan
        file_name=dirs+"/"+folder_name
        for files in os.listdir(file_name):     #15.csv 16.csv
            I_raw_data=pd.read_csv(file_name + "/" + files, skiprows=None, nrows=None,header=None,na_filter=False)
            data.append(I_raw_data.T)
            labels.append(name_dict[folder_name])
    data=np.array(data)
    labels=np.array(labels)
    
    return data,labels
def load_totle_data(image_dirs,dirs=None):
    image_path=[]
    labels=[]
    data=[]
    name_dict={'Air Conditioner':0,'Compact Fluorescent Lamp':1,'Fan':2,'Fridge':3,'Hairdryer':4,'Heater':5,'Incandescent Light Bulb':6,'Laptop':7,'Microwave':8,'Vacuum':9,'Washing Machine':10}
    for folder_name in os.listdir(dirs):        #Air Conditioner,Fan
        file_name=dirs+"/"+folder_name
        for files in os.listdir(file_name):
            I_raw_data=pd.read_csv(file_name + "/" + files, skiprows=None, nrows=None,header=None,na_filter=False)
            file_num=files.split(".")[0]
            cla_path=image_dirs+"/"+folder_name+"/"+file_num+".png"
            data.append(I_raw_data.T)
            labels.append(name_dict[folder_name])
            image_path.append(cla_path)
    data=np.array(data)
    labels=np.array(labels)
    image_path=np.array(image_path)
    return data,labels,image_path
def load_image_path(image_dirs,dirs=None):
    image_path=[]
    for folder_name in os.listdir(dirs):        #Air Conditioner,Fan
        file_name=dirs+"/"+folder_name
        for files in os.listdir(file_name):
            files=files.split(".")[0]
            cla_path=image_dirs+"/"+folder_name+"/"+files+".png"
            image_path.append(cla_path)
    return np.array(image_path)
def load(config,X_train=None,y_train=None,X_test=None,y_test=None,images_train=None,images_test_data=None):
    Data={}
    problem = config['data_dir'].split('/')[-1]                 #Current

    if os.path.exists(config['data_dir'] + '/' + problem + '.npy'):
        '''
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + problem + '.npy', allow_pickle=True)

        Data['max_len'] = Data_npy.item().get('max_len')
        Data['All_train_data'] = Data_npy.item().get('All_train_data')
        Data['All_train_label'] = Data_npy.item().get('All_train_label')
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['val_data'] = Data_npy.item().get('val_data')
        Data['val_label'] = Data_npy.item().get('val_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')
        Data['image_train_data']=Data_npy.item().get('image_train_data')
        Data['image_val_data']=Data_npy.item().get('image_val_data')
        Data['images_test_data']=Data_npy.item().get('images_test_data')
        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))
        '''
    else:
        logger.info("Loading and preprocessing data ...")
        train_file = config['data_dir'] + "/train"
        test_file = config['data_dir'] + "/test"
        image_dirs = "data/images11/vmdwpt_db6_L3_imif_final"
        if X_train is None:
            X_train, y_train = load_from_csv(train_file)       #/dataset/NILM/Current/train
            X_test, y_test = load_from_csv(test_file)
            images_train=load_image_path(image_dirs,train_file)
            images_test_data=load_image_path(image_dirs,test_file)

        max_seq_len=X_train.shape[2]                       #以后可能会变动

        if config['Norm']:
            mean, std = mean_std(X_train)
            mean = np.repeat(mean, max_seq_len).reshape(X_train.shape[1], max_seq_len)
            std = np.repeat(std, max_seq_len).reshape(X_train.shape[1], max_seq_len)
            X_train = mean_std_transform(X_train, mean, std)
            X_test = mean_std_transform(X_test, mean, std)
        
        Data['max_len'] = max_seq_len
        Data['All_train_data'] = X_train
        Data['All_train_label'] = y_train

        if config['val_ratio'] > 0:
            train_data, train_label, val_data, val_label,image_train_data,image_val_data = split_dataset(images_train,X_train, y_train, config['val_ratio'])
        else:
            val_data, val_label = [None, None]

        logger.info("{} samples will be used for training".format(len(train_label)))
        logger.info("{} samples will be used for validation".format(len(val_label)))
        logger.info("{} samples will be used for testing".format(len(y_test)))
        Data['train_data'] = train_data
        Data['train_label'] = train_label
        Data['val_data'] = val_data
        Data['val_label'] = val_label
        Data['test_data'] = X_test
        Data['test_label'] = y_test
        Data['image_train_data']=image_train_data
        Data['image_val_data']=image_val_data
        Data['images_test_data']=images_test_data

        # np.save(config['data_dir'] + "/" + problem, Data, allow_pickle=True)

    return Data

def mean_std(train_data):
    m_len = np.mean(train_data, axis=2)
    mean = np.mean(m_len, axis=0)

    s_len = np.std(train_data, axis=2)
    std = np.max(s_len, axis=0)

    return mean, std


def mean_std_transform(train_data, mean, std):
    return (train_data - mean) / std

def split_dataset(image_train, data, label, validation_ratio):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    image_train_data=image_train[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    image_val_data=image_train[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label,image_train_data,image_val_data