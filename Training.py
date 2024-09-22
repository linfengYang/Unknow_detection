import os
import logging
import torch
import numpy as np
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from Models.loss import l2_reg_loss
from Models import utils, analysis
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from pyod.models.lunar import LUNAR         
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from numpy import percentile
# from adbench.baseline.DAGMM.run import DAGMM
# from adbench.baseline.PyOD import PYOD
# from adbench.baseline.DeepSAD.src.run import DeepSAD
# from adbench.baseline.REPEN.run import REPEN
# from adbench.baseline.FEAWAD.run import FEAWAD
# from adbench.baseline.PReNet.run import PReNet
# from adbench.baseline.GANomaly.run import GANomaly
logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


class BaseTrainer(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat =False):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class SupervisedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):

        super(SupervisedTrainer, self).__init__(*args, **kwargs)

        if isinstance(args[3], torch.nn.CrossEntropyLoss):
            # self.classification = True  # True if classification, False if regression
            self.analyzer = analysis.Analyzer(print_conf_mat=False)
        else:
            self.classification = False
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):

            image_data, X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device),image_data.to(self.device))         #待修改，两段向量

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization
            # if self.l2_reg:   改为 if self.l2_reg != 0:
            
            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss
            
            total_loss = mean_loss + 0.5 * l2_reg_loss(self.model)
            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            '''
            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)
            '''
            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):
            image_data, X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device),image_data.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            #if i % self.print_interval == 0:
                #ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                #self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions,dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes
        # self.epoch_metrics['f1'] = metrics_dict['f1']
        '''
        if self.model.num_classes == 2:
            false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])  # 1D scores needed
            self.epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
            self.epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)
        '''
        return self.epoch_metrics, metrics_dict


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    #logger.info("Evaluating on validation set ...")
    #eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, ConfMat = val_evaluator.evaluate(epoch, keep_all=True)
    #eval_runtime = time.time() - eval_start_time
    #logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    #global val_times
   # val_times["total_time"] += eval_runtime
    #val_times["count"] += 1
    #avg_val_time = val_times["total_time"] / val_times["count"]
    #avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    #avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    #logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
    #logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    #logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = 'Validation Summary: '
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model) 
        best_metrics = aggr_metrics.copy()

        #pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        # np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value


def train_runner(config, model, trainer, val_evaluator, path):
    epochs = config['epochs']
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    save_best_model = utils.SaveBestModel()
    # save_best_acc_model = utils.SaveBestACCModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                              best_value, epoch)
        save_best_model(aggr_metrics_val['loss'], epoch, model, optimizer, loss_module, path)
        #save_best_acc_model(aggr_metrics_val['accuracy'], epoch, model, optimizer, loss_module, path)

        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return

def train_unknow(best_model,train_loader,val_loader,test_loader,device):

    best_model=best_model.eval()
    per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
    for i, batch in enumerate(test_loader):
        image_data, X, targets, IDs = batch
        targets = targets.to(device)
        predictions = best_model(X.to(device),image_data.to(device),False)
        # predictions = best_model(X.to(device),image_data.to(device),True)
        per_batch['targets'].append(targets.cpu().numpy())
        predictions = predictions.detach()
        per_batch['predictions'].append(predictions.cpu().numpy())
    predictions = np.concatenate(per_batch['predictions'], axis=0)
    targets = np.concatenate(per_batch['targets'], axis=0).flatten()

    np.savez("test_data.npz",predictions,targets)
    per_batch1 = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
    for i, batch in enumerate(train_loader):
        image_data, X, targets, IDs = batch
        targets = targets.to(device)
        predictions = best_model(X.to(device),image_data.to(device),False)
        # predictions = best_model(X.to(device),image_data.to(device),True)
        per_batch1['targets'].append(targets.cpu().numpy())
        predictions = predictions.detach()
        per_batch1['predictions'].append(predictions.cpu().numpy())
    predictions1 = np.concatenate(per_batch1['predictions'], axis=0)
    targets1 = np.concatenate(per_batch1['targets'], axis=0).flatten()

    np.savez("train_data.npz",predictions1,targets1)
    per_batch2 = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
    for i, batch in enumerate(val_loader):
        image_data, X, targets, IDs = batch
        targets = targets.to(device)
        predictions = best_model(X.to(device),image_data.to(device),False)
        # predictions = best_model(X.to(device),image_data.to(device),True)
        per_batch2['targets'].append(targets.cpu().numpy())
        predictions = predictions.detach()
        per_batch2['predictions'].append(predictions.cpu().numpy())
    predictions2 = np.concatenate(per_batch2['predictions'], axis=0)
    targets2 = np.concatenate(per_batch2['targets'], axis=0).flatten()

    np.savez("val_data.npz",predictions2,targets2)

    X_train=np.concatenate([predictions1,predictions2],0)
    y_train=np.concatenate([targets1,targets2],0)

    return X_train,y_train

def unknow_module(unknow_point,cnt,X_train=None,y_train=None,X_test=None,y_test=None,best_model=None,test_loader=None,device=None):
    #加载数据集，如果已经有就直接加载（为了方便跟没有双通道对比）
    if X_test is None:
        X_train=X_train
        y_train=y_train
        # r=np.load("train_data.npz")
        # X_train,y_train=r['arr_0'],r['arr_1']

        # r=np.load("test_data.npz")
        # X_test,y_test=r['arr_0'],r['arr_1']

        # r=np.load("val_data.npz")
        # X_val,y_val=r['arr_0'],r['arr_1']

        # X_train=np.concatenate([X_train,X_val],0)
        # y_train=np.concatenate([y_train,y_val],0)
    else:
        X_train=np.squeeze(X_train,1)
        X_test=np.squeeze(X_test,1)

    
    #测试softmax出来的概率效果
    per_batch1 = {'targets': [], 'predictions': [],'predictions_pre':[]}
    for i, batch in enumerate(test_loader):
        image_data, X, targets, IDs = batch
        targets = targets.to(device)
        predictions_pre = best_model(X.to(device),image_data.to(device),False)          #相当于在前一步移出来，32+32=64维度，不经过softmax
        predictions = best_model(X.to(device),image_data.to(device),True)
        per_batch1['targets'].append(targets.cpu().numpy())
        predictions_pre = predictions_pre.detach()
        predictions = predictions.detach()
        per_batch1['predictions_pre'].append(predictions_pre.cpu().numpy())
        per_batch1['predictions'].append(predictions.cpu().numpy())
    predictions_pre = np.concatenate(per_batch1['predictions_pre'], axis=0)
    predictions = np.concatenate(per_batch1['predictions'], axis=0)
    targets1 = np.concatenate(per_batch1['targets'], axis=0).flatten()
    predictions = torch.from_numpy(predictions)
    probs = torch.nn.functional.softmax(predictions,dim=1)
    probs1=torch.argmax(probs,axis=1).cpu().numpy()
    probs2 = torch.amax(probs,axis=1).cpu().numpy()
    # np.savetxt('data.csv', probs, delimiter=',',fmt="%.4f")
    # print("vector_shape:{}".format(predictions_pre.shape))

    X_test=predictions_pre
    y_test=targets1
    
    #修改为当前01数据
    yy_train=np.zeros((y_train.shape[0]))
    yy_test=np.zeros((y_test.shape[0]),dtype=int)
    for j in range(y_test.shape[0]):
        if y_test[j]==unknow_point:
        # if y_test[j] in unknow_point:     #多电器
            yy_test[j]=1
    # yy_train[0]=1
    # clf_name = 'DeepSVDD'
    # clf = DeepSVDD(use_ae=False, epochs=20, contamination=0,
                #    random_state=10)
    

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    
    clf_name="LUNAR"
    # clf = DeepSVDD(use_ae=True, epochs=20, contamination=0.25,
    #                random_state=20)
    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                 LOF(n_neighbors=25), LOF(n_neighbors=35),
                 COPOD()]
    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                     LOF(n_neighbors=25), LOF(n_neighbors=35)]
    #SUBSPACE  UNIFORM
    # clf = LUNAR(n_neighbours=12,n_epochs=500,contamination=0.10,proportion=0.10)     #k=12 阈值0.9，0.85适合plaid2017
    clf = LUNAR(n_neighbours=7,n_epochs=500,contamination=0.02,proportion=0.10)        #k=7 阈值0.98，0.95适合whited
    # clf=DeepSVDD()
    # clf=LOF()
    clf.fit(X_train)

    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    
    TP=0
    FP=0
    FN=0
    TN=0
    # print("-----------------predict result----------------")
    # print(y_test_pred.shape)
    # print("-----------------true result(0,1)--------------")
    # print(yy_test.shape)
    # np.savetxt('data.csv', np.stack([probs2,probs1,yy_test,y_test_pred],axis=0).T, delimiter=',',fmt="%.4f",header="softmax,true,predict")    #softmax,真实，预测
    for k in range(yy_test.shape[0]):
        if yy_test[k]==1 and yy_test[k]==y_test_pred[k]:
            TP+=1
        if yy_test[k]==1 and yy_test[k]!=y_test_pred[k]:
            FN+=1
        if yy_test[k]==y_test_pred[k] and yy_test[k]==0:
            TN+=1
        if yy_test[k]!=y_test_pred[k] and yy_test[k]==0:
            FP+=1
    print("TP:{} FP:{} FN:{} TN:{}".format(TP,FP,FN,TN))
    print(1-FP/yy_test.shape[0])
    print(TP/sum(yy_test))
    
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    # print("\nOn Training Data:")
    # evaluate_print(clf_name, yy_train, y_train_scores)
    print("On Test Data:")
    evaluate_print(clf_name, yy_test, y_test_scores)
    f_score=np.round(f1_score(y_true=yy_test, y_pred=y_test_pred, average='binary'),decimals=4)
    roc=np.round(roc_auc_score(yy_test, y_test_scores), decimals=4)
    print("f1score:{}".format(f_score))

    #绘制f1score图和混淆矩阵图(为了位置类别)
    #把预测出的(0,1)->(已知，未知)中的1标签位置，全部改为未知类别
    for i in range(y_test_pred.shape[0]):
        if y_test_pred[i]==1:
            probs1[i]=unknow_point
            # probs1[i]=unknow_point[0]
    analyser=analysis.Analyzer(print_conf_mat=True)
    metrics_dict=analyser.analyze_classification(probs1,targets1,np.arange(12))

    np.savez("space_plot.npz",predictions_pre,targets1)
    # metrics_dict['f1'][unknow_point[0]]=f_score                                 #多电器使用

    print(np.round(metrics_dict['f1'],decimals=4))
    f_macro=np.round(sum(metrics_dict['f1'])/len(metrics_dict['f1']),decimals=4)
    print(f_macro)
    #图片展示测试
    ts=TSNE(n_components=2,init='pca',random_state=3407)
    X_train=ts.fit_transform(X_train)
    X_test=ts.fit_transform(X_test)
    x_min, x_max = np.min(X_test, 0), np.max(X_test, 0)
    X_test = (X_test - x_min) / (x_max - x_min)
    visualize(clf_name, X_train, yy_train, X_test, yy_test, y_train_pred,y_test_pred, show_figure=False, save_figure=True)
    #lunar等高线图片
    # y_test_scores=y_test_scores*-1
    # threshold = percentile(y_test_scores, 98)
    xx,yy=np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
    X1 = 0.3 * np.random.randn(160 // 2, 2)
    X2 = 0.3 * np.random.randn(160 // 2, 2)
    X = np.r_[X1, X2]
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(40, 2))]
    clf=LUNAR()
    clf.fit(X)
    y_test_scores=clf.decision_function(X)*-1
    threshold = percentile(y_test_scores, 21)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
    fig=plt.figure(figsize=(12,5))
    subplot1=fig.add_subplot(121)
    subplot=fig.add_subplot(122)
    # fig.suptitle('Horizontally stacked subplots')
    # subplot=plt.subplot(111)
    # subplot1=plt.subplot()
    subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
		# a = subplot.contour(xx, yy, Z, levels=[Z.min(),threshold],
		                    # linewidths=2, colors='red')
    subplot.contourf(xx, yy, Z, levels=[threshold,Z.max()],
						 colors='orange')
    b = subplot.scatter(X[0:-40, 0], X[0:-40, 1], c='white',
							s=10,edgecolors='k')
    c = subplot.scatter(X[-40:, 0], X[-40:, 1], c='black',
							s=10,edgecolors='k')
    b = subplot1.scatter(X[0:-40, 0], X[0:-40, 1], c='white',
							s=10,edgecolors='k')
    c = subplot1.scatter(X[-40:, 0], X[-40:, 1], c='black',
							s=10,edgecolors='k')
    subplot.legend(
        [b, c],
        ['inliers', 'outliers'],
        loc='lower right')
    subplot1.legend(
        [b, c],
        ['inliers', 'outliers'],
        loc='lower right')
    plt.savefig('ALL.png', dpi=300, bbox_inches='tight')



    with open("unknow_111.txt","a")as f:
        f.write("the {}: know_precision: {} unknow_precision: {} f1_score:{} auroc:{} f_macro:{} \n".format(cnt,(1-FP/yy_test.shape[0]),(TP/sum(yy_test)),f_score,roc,f_macro))
    
    
    # clf_name="DeeSAD"
    # model = GANomaly(seed=42)
    # model.fit(X_train, yy_train)  # fit
    # score = model.predict_score(X_test)
    # evaluate_print(clf_name,yy_test,score)
    # print(len(score))



