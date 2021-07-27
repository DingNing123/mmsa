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
import wandb


class TFN():
    def __init__(self, args):
        assert args.tasks in ['M']

        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop().getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam(list(model.parameters())[2:], lr=self.args.learning_rate)
        # initilize results
        best_acc = 0
        epochs, best_epoch = 0, 0

        # 可视化
        os.environ["WANDB_PROGRAM"] = "prepare_mustard_run_oursplit.py"
        wandb.init(project='mustard', entity='dingning')

        args = self.args
        wandb.config.update(args)
        # wandb.config.update(args, allow_val_change=True)

        test_accuracies = []
        test_f_scores = []
        test_maes = []
        test_corrs = []
        # 可视化

        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            # with tqdm(dataloader['train']) as td:
            for batch_data in dataloader['train']:
                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                labels = batch_data['labels'][self.args.tasks].to(self.args.device).view(-1, 1)
                # clear gradient
                optimizer.zero_grad()
                # forward
                outputs = model(text, audio, vision)
                # compute loss
                loss = self.criterion(outputs[self.args.tasks], labels)
                # backward
                loss.backward()
                # update
                optimizer.step()
                # store results
                train_loss += loss.item()
                y_pred.append(outputs[self.args.tasks].cpu())
                y_true.append(labels.cpu())
            train_loss = train_loss / len(dataloader['train'])
            print("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, train_loss))
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            print('%s: >> ' %(self.args.tasks) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            val_acc = val_results[self.args.tasks][self.args.KeyEval]

            # 可视化
            # val_results[self.args.tasks].keys()
            # dict_keys(['Mult_acc_2', 'Mult_acc_3', 'Mult_acc_5', 'F1_score', 'recall', 'MAE', 'Corr'])
            # train_results.keys()
            # dict_keys(['Mult_acc_2', 'Mult_acc_3', 'Mult_acc_5', 'F1_score', 'recall', 'MAE', 'Corr'])

            train_acc = train_results['Mult_acc_2']
            train_f_score = train_results['F1_score']

            test_acc = val_results[self.args.tasks]['Mult_acc_2']
            test_mae = val_results[self.args.tasks]['MAE']
            test_corr = val_results[self.args.tasks]['Corr']
            test_f_score = val_results[self.args.tasks]['F1_score']

            test_accuracies.append(test_acc)
            test_f_scores.append(test_f_score)
            test_maes.append(test_mae)
            test_corrs.append(test_corr)

            wandb.log(
                # (
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f_score": train_f_score,

                    "test_acc": test_acc,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "test_f_score": test_f_score,

                    "best_test_acc": max(test_accuracies),
                    "best_f1": max(test_f_scores),
                    "best_mae": min(test_maes),
                    "best_corr": max(test_corrs),

                    "best_epoch": test_f_scores.index(max(test_f_scores)),
                    'epoch': epochs

                }
                # )
            )

            # 可视化

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
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            # with tqdm(dataloader) as td:
            for batch_data in dataloader:
                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                labels = batch_data['labels'][self.args.tasks].to(self.args.device).view(-1, 1)
                outputs = model(text, audio, vision)
                loss = self.criterion(outputs[self.args.tasks], labels)
                eval_loss += loss.item()
                y_pred.append(outputs[self.args.tasks].cpu())
                y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        print(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        print('%s: >> ' %(self.args.tasks) + dict_to_str(results))
        tmp = {
            self.args.tasks: results
        }
        return tmp