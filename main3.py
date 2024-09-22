import os
import argparse
import logging
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from art import *
# Import Project Modules -----------------------------------------------------------------------------------------------
from utils_bk import Setup, Initialization, Data_Loader, dataset_class, Data_Verifier
from Models.model import model_factory, count_parameters
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Models.utils import load_model
from Training import SupervisedTrainer, train_runner,train_unknow,unknow_module
from torchvision import transforms
from sklearn.model_selection import KFold
from Dataset.load_nilm_data import load_from_csv,load_whited_from_csv
from Dataset.load_nilm_data import load_image_path
from sklearn.model_selection import StratifiedKFold
from Dataset.load_nilm_data import load_totle_data

from datetime import datetime
logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='Dataset/NILM', choices={'Dataset/UEA/', 'Dataset/Segmentation/','Dataset/NILM/'},
                    help='Data path')
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
# 0.2
parser.add_argument('--val_ratio', type=float, default=0.1, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Net_Type', default=['1D-2D'], choices={'T', 'C-T','1D-2D'}, help="Network Architecture. Convolution (C)"
                                                                              "Transformers (T)") #C-T
                                                                              
# Transformers Parameters ------------------------------
parser.add_argument('--emb_size', type=int, default=32, help='Internal dimension of transformer embeddings') # 16
parser.add_argument('--dim_ff', type=int, default=512, help='Dimension of dense feedforward part of transformer layer') # 256
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads') #8
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'Sin','None'}, default='None',
                    help='Fix Position Embedding')
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector','attention_ROPE','None'}, default='attention_ROPE',
                    help='Relative Position Embedding')
# Training Parameters/ Hyper-Parameters ----------------
parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')        #batch_size100
parser.add_argument('--batch_size', type=int, default=10, help='Training batch size')  #16
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') #1e-3 #5e-3
parser.add_argument('--dropout', type=float, default=0.02, help='Dropout regularization ratio')    #0.02
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
args = parser.parse_args()

if __name__ == '__main__':
    config = Setup(args)  # configuration dictionary
    device = Initialization(config)
    Data_Verifier(config)  # Download the UEA and HAR datasets if they are not in the directory
    All_Results = ['Datasets', 'ConvTran']  # Use to store the accuracy of ConvTran in e.g "Result/Datasets/UEA"

    for problem in os.listdir(config['data_path']):  # for loop on the all datasets in "data_dir" directory
        config['data_dir'] = os.path.join(config['data_path'], problem)     #   /Dataset/UEA/Multivariate_ts  or /Dataset/NILM/Current
        # print(config['data_dir'])
        print(text2art(problem, font='small'))
        # ------------------------------------ Load Data ---------------------------------------------------------------
        logger.info("Loading Data ...")

        # Data = Data_Loader(config)              # 8:2实验
        
        result_finally=[]                                                   #储存每一轮结果，便于计算
        acc=[]
        preci=[]
        
        # train_file=config['data_dir']+"/totle_data"
        # image_file="data/images11/vmdwpt_db6_L3_imif_final"
        # # X_train,y_train=load_from_csv(train_file)
        # # image_totle_path=load_image_path(image_file,train_file)
        # X_totle,y_totle,image_totle_path=load_totle_data(image_file,train_file)
        # kfold=StratifiedKFold(n_splits=10,shuffle=True)
        # result_finally=[]                                                   #储存每一轮结果，便于计算
        # acc=[]
        # preci=[]
        # for train_index,test_index in kfold.split(X_totle,y_totle):
        #     this_train_x,this_test_x=X_totle[train_index],X_totle[test_index]
        #     this_train_y,this_test_y=y_totle[train_index],y_totle[test_index]
        #     image_train_path,image_test_path=image_totle_path[train_index],image_totle_path[test_index]
        #     Data=Data_Loader(config,this_train_x,this_train_y,this_test_x,this_test_y,image_train_path,image_test_path)
        
        # train_file = config['data_dir'] + "/train"
        # test_file = config['data_dir'] + "/test"
        # image_dirs = "data/images11/vmdwpt_db6_L3_imif_final"
        # XX_train, yy_train = load_from_csv(train_file)       #/dataset/NILM/Current/train
        # XX_test, yy_test = load_from_csv(test_file)
        # images_train=load_image_path(image_dirs,train_file)
        # images_test_data=load_image_path(image_dirs,test_file)
        train_file = config['data_dir'] + "/train_whited"
        test_file = config['data_dir'] + "/test_whited"
        image_dirs = "data/whited_image2/vmdwpt_db6_L3_imif_final"
        XX_train, yy_train = load_whited_from_csv(train_file)       #/dataset/NILM/Current/train
        XX_test, yy_test = load_whited_from_csv(test_file)
        images_train=load_image_path(image_dirs,train_file)
        images_test_data=load_image_path(image_dirs,test_file)
        unknows_Plaid2017_list=[[0,5,9],[1,2,9],[3,4,8],[6,7,9],[2,8,10]]
        unknows_white_list=[[0,1,9],[1,3,6],[4,5,8],[2,7,11],[4,6,10]]
        cnt=0
        for i in (range(11)):
        # for i in unknows_Plaid2017_list:
        # for i in unknows_white_list:
            # i=[1,2,9]
            # i=1
            # i=3
            # i=unknows_Plaid2017_list[3]
            i=0
            cnt+=1
            X_train=[]
            y_train=[]
            images_train_per=[]
            X_test=[]
            y_test=[]
            images_test_per=[]
            for j in range(yy_train.shape[0]):
                if yy_train[j]!=i:
                # if yy_train[j] not in i:
                    X_train.append(XX_train[j])
                    y_train.append(yy_train[j])
                    images_train_per.append(images_train[j])
                # if yy_train[j]==i:
                else:
                    X_test.append(XX_train[j])
                    y_test.append(yy_train[j])
                    images_test_per.append(images_train[j])
            X_train=np.array(X_train)
            y_train=np.array(y_train)
            images_train_per=np.array(images_train_per)
            X_test=np.array(X_test)
            y_test=np.array(y_test)
            images_test_per=np.array(images_test_per)
            print(X_test.shape)
            print(XX_test.shape)
            X_test=np.concatenate((XX_test,X_test),axis=0)
            y_test=np.concatenate((yy_test,y_test),axis=0)
            np.savez("origin.npz",X_test.squeeze(1),y_test)
            images_test_per=np.concatenate((images_test_data,images_test_per),axis=0)
            Data=Data_Loader(config,X_train,y_train,X_test,y_test,images_train_per,images_test_per)

            print(type(Data['train_data']))
            print(Data['test_data'].shape)
            print(Data['image_train_data'].shape)
            img_size = 64
            data_transform = {
                "train": transforms.Compose([transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]),
                "val": transforms.Compose([
                                        transforms.Resize(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])}
            train_dataset = dataset_class(Data['image_train_data'], Data['train_data'], Data['train_label'],data_transform['train'])
            val_dataset = dataset_class(Data['image_val_data'], Data['val_data'],Data['val_label'],data_transform['val'])
            test_dataset = dataset_class(Data['images_test_data'],Data['test_data'],Data['test_label'],data_transform['train'])

            train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
            # --------------------------------------------------------------------------------------------------------------
            # -------------------------------------------- Build Model -----------------------------------------------------
            dic_position_results = [config['data_dir'].split('/')[-1]]

            logger.info("Creating model ...")
            config['Data_shape'] = Data['train_data'].shape
            config['num_labels'] = int(max(Data['train_label']))+1
            model = model_factory(config)
            # print(model.state_dict())
            # logger.info("Model:\n{}".format(model))
            logger.info("Total number of parameters: {}".format(count_parameters(model)))
            # -------------------------------------------- Model Initialization ------------------------------------
            optim_class = get_optimizer("Adam")  #RAdam,Adam
            config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
            config['loss_module'] = get_loss_module()
            save_path = os.path.join(config['save_dir'], problem + 'model_{}.pth'.format('last'))
            tensorboard_writer = SummaryWriter('summary')
            model.to(device)
            # ---------------------------------------------- Training The Model ------------------------------------
            logger.info('Starting training...')
            #l2_reg=0，两段向量修改todo
            trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0,
                                        print_interval=config['print_interval'], console=config['console'], print_conf_mat=False)
            val_evaluator = SupervisedTrainer(model, val_loader, device, config['loss_module'],
                                            print_interval=config['print_interval'], console=config['console'],
                                            print_conf_mat=False)

            train_runner(config, model, trainer, val_evaluator, save_path)

            best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
            best_model.to(device)

            X_train,y_train=train_unknow(best_model,train_loader,val_loader,test_loader,device)
            # unknow_module(unknow_point=i,cnt=cnt,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
            unknow_module(unknow_point=i,X_train=X_train,y_train=y_train,cnt=cnt,best_model=best_model,test_loader=test_loader,device=device)
            exit()
            '''
            best_test_evaluator = SupervisedTrainer(best_model, test_loader, device, config['loss_module'],
                                                    print_interval=config['print_interval'], console=config['console'],
                                                    print_conf_mat=True)
            best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
            print_str = 'Best Model Test Summary: '
            for k, v in best_aggr_metrics_test.items():
                print_str += '{}: {} | '.format(k, v)
                if k=='accuracy':
                    acc.append(v)
                if k=='precision':
                    preci.append(v)
            print(print_str)
            result_finally.append(print_str)
            dic_position_results.append(all_metrics['total_accuracy'])
            problem_df = pd.DataFrame(dic_position_results)
            problem_df.to_csv(os.path.join(config['pred_dir'] + '/' + problem + '.csv'))

            All_Results = np.vstack((All_Results, dic_position_results))
        print(result_finally)
        accuracy=sum(acc)/len(acc)
        precision=sum(preci)/len(acc)
        print(accuracy)
        print(precision)
        str=datetime.now().strftime("%Y-%m-%d_%H-%M")
        with open(str+".txt","w")as f:
            f.write("accuracy:{}|precision:{}".format(accuracy,precision))
    All_Results_df = pd.DataFrame(All_Results)
    All_Results_df.to_csv(os.path.join(config['output_dir'], 'ConvTran_Results.csv'))
    '''
