"""
TRAINS/MULTITASK/MLFDNN.PY DN 2020

"""

import os
import time
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

class MLF_DNN():
    def __init__(self, args):
        print('construct train MLF_DNN class  ')
        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop().getMetics(args.datasetName)
        # DN

    def do_train(self, model, dataloader):
        print('inf file train MLF_DNN.py  do_train  '
              '学习率 MLLF_DNN:',self.args.learning_rate)

        print('model.Model.audio_subnet')
        print(model.Model.audio_subnet)

        print('model.Model')
        print(model.Model)

        optimizer = optim.Adam([{"params": list(model.Model.text_subnet.parameters()), "weight_decay": self.args.text_weight_decay},
                                {"params": list(model.Model.audio_subnet.parameters()), "weight_decay": self.args.audio_weight_decay},
                                {"params": list(model.Model.video_subnet.parameters()), "weight_decay": self.args.video_weight_decay}],
                                lr=self.args.learning_rate)
        # initilize results
        best_acc = 0
        epochs, best_epoch = 0, 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            print('dn test dataloader[train] ', dataloader['train'])
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    print('labels.keys(): ')
                    print(labels.keys())
                    for k in labels.keys():
                        labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    print('text shape:', text.shape)
                    print('audio shape:', audio.shape)
                    print('vision shape:', vision.shape)
                    print('labels:',labels)
                    outputs = model(text, audio, vision)
                    # compute loss
                    loss = 0.0
                    print('the loss are accumulated .... ')
                    for m in self.args.tasks:
                        loss += eval('self.args.'+m) * self.criterion(outputs[m], labels[m])
                        # the paper in table 4 use M Label it is right  yes ....
                        # loss += eval('self.args.'+m) * self.criterion(outputs[m], labels['M'])
                        print("'self.args.'+m")
                        print('=======self.args.==========='+m)
                        print()
                        print('there is a rate : 0.8 0.6 0.4 0.2 ')
                        print("eval('self.args.'+m)")
                        print(eval('self.args.'+m))
                    print('completed  the loss are accumulated .... ')
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    for m in self.args.tasks:
                        print('m: ')
                        print(m)
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
            train_loss = train_loss / len(dataloader['train'])
            print("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, train_loss))



            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                print('train_results: in different tasks' , m )
                print('%s: >>  ' %(m) + dict_to_str(train_results))


            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            val_acc = val_results[self.args.tasks[0]][self.args.KeyEval]
            # save best model
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epochs
                model_path = os.path.join(self.args.model_save_path,\
                                    f'{self.args.modelName}-{self.args.datasetName}-{self.args.tasks}.pth')
                if os.path.exists(model_path):
                    os.remove(model_path)
                # save model
                torch.save(model.cpu().state_dict(), model_path)
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    outputs = model(text, audio, vision)
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += eval('self.args.'+m) * self.criterion(outputs[m], labels[m])
                    eval_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
        eval_loss = eval_loss / len(dataloader)
        print(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        return_res = {}
        for m in self.args.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            print('%s: >> ' %(m) + dict_to_str(results))
            return_res[m] = results
        return return_res