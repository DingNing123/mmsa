import os
import time
import random
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config_run import Config
from config.config_debug import ConfigDebug
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', type=bool, default=False,
                        help='adjust parameters ?')
    parser.add_argument('--modelName', type=str, default='mlf_dnn',
                        help='support mult/tfn/lmf/mfn/ef_lstm/lf_dnn/mtfn/mlmf/mlf_dnn')
    parser.add_argument('--datasetName', type=str, default='sims',
                        help='support mosi/sims/mustard')
    parser.add_argument('--tasks', type=str, default='MT',
                        help='M/T/A/V/MTAV/...')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_path', type=str, default='results/model_saves',
                        help='path to save model.')
    parser.add_argument('--res_save_path', type=str, default='results/result_saves',
                        help='path to save results.')
    parser.add_argument('--data_dir', type=str, default='/media/dn/85E803050C839C68/datasets/',
                        help='path to data directory')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used.')
    return parser.parse_args()

args = parse_args()
data_path = args.data_dir + 'CH-SIMS/Processed/features/data.npz.npz'
data = np.load(data_path)
print()