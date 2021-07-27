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


class LMF():
    def __init__(self, args):
        assert args.tasks in ['M']

        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop().getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam([{"params": list(model.parameters())[:3], "lr": self.args.factor_lr},
                                {"params": list(model.parameters())[5:], "lr": self.args.learning_rate}],
                                weight_decay=self.args.weight_decay)
        # initilize results
        best_acc = 0
        epochs, best_epoch = 0, 0

        os.environ["WANDB_PROGRAM"] = "run.py"
        wandb.init(project='mustard', entity='dingning')
        args = self.args
        wandb.config.update(args, allow_val_change=True)

        test_accuracies = []
        test_f_scores = []
        test_maes = []
        test_corrs = []
        # 可视化

        while True: 
            epochs += 1
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            for batch_data in dataloader['train']:
                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                labels = batch_data['labels'][self.args.tasks].to(self.args.device).view(-1, 1)
                optimizer.zero_grad()
                outputs = model(text, audio, vision)
                loss = self.criterion(outputs[self.args.tasks], labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                y_pred.append(outputs[self.args.tasks].cpu())
                y_true.append(labels.cpu())
            train_loss = train_loss / len(dataloader['train'])

            print()
            print('-'*10,'epochs:',epochs)
            print("train loss: %.4f " % (train_loss))
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            precision = train_results['precision']
            recall = train_results['recall']
            f1_score = train_results['f1_score']
            print(f"precision:{precision:.4f} recall:{ recall:.4f} f1_score:{ f1_score:.4f}")

            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="val")
            val_acc = val_results[self.args.tasks]['f1_score']
            test_acc = val_results[self.args.tasks]['precision']
            # test_mae = val_results[self.args.tasks]['MAE']
            # test_corr = val_results[self.args.tasks]['Corr']
            test_f_score = val_results[self.args.tasks]['f1_score']
            test_accuracies.append(test_acc)
            test_f_scores.append(test_f_score)
            # test_maes.append(test_mae)
            # test_corrs.append(test_corr)

            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": precision ,
                    "train_f_score": f1_score ,

                    "test_acc": test_acc,
                    # "test_mae": test_mae,
                    # "test_corr": test_corr,
                    "test_f_score": test_f_score,

                    "best_test_acc": max(test_accuracies),
                    "best_f1": max(test_f_scores),
                    # "best_mae": min(test_maes),
                    # "best_corr": max(test_corrs),

                    "best_epoch": test_f_scores.index(max(test_f_scores)),
                    'epoch': epochs
                }
            )
            # save best model
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epochs
                model_path = os.path.join(self.args.model_save_path, \
                                          f'{self.args.modelName}-{self.args.datasetName}-{self.args.tasks}.pth')
                if os.path.exists(model_path):
                    os.remove(model_path)
                torch.save(model.cpu().state_dict(), model_path)
                model.to(self.args.device)
            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_test(self, model, dataloader, mode="val"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
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

        print()
        print(mode + " loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        val_results = self.metrics(pred, true)
        precision = val_results['precision']
        recall = val_results['recall']
        f1_score = val_results['f1_score']
        print(f"precision:{precision:.4f} recall:{ recall:.4f} f1_score:{ f1_score:.4f}")
        tmp = {
            self.args.tasks: val_results
        }

        return tmp
