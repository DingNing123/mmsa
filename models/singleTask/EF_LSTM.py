"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.subNets.FeatureNets import SubNet, TextSubNet

__all__ = ['EF_LSTM']

class EF_LSTM(nn.Module):
    """
    early fusion using lstm
    """
    def __init__(self, args):
        super(EF_LSTM, self).__init__()
        self.args = args
        in_size = self.get_insize()
        # import pdb;pdb.set_trace()
        input_len = args.input_lens
        hidden_size = args.hidden_dims
        num_layers = args.num_layers
        dropout = args.dropout
        output_dim = args.output_dim 
        self.norm = nn.BatchNorm1d(input_len)
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)

    def get_insize(self):
        text_in, audio_in, video_in = self.args.feature_dims
        if self.args.modality == 't':
            in_size = text_in
        if self.args.modality == 'a':
            in_size = audio_in
        if self.args.modality == 'v':
            in_size = video_in 
        if self.args.modality == 'tav':
            in_size = text_in + audio_in + video_in
        if self.args.modality == 'ta':
            in_size = text_in + audio_in 
        if self.args.modality == 'tv':
            in_size = text_in + video_in 
        if self.args.modality == 'av':
            in_size = audio_in + video_in

        return in_size


    def forward(self, text_x, audio_x, video_x):
        if self.args.modality == 't':
            x = torch.cat([text_x, ], dim=-1)
        if self.args.modality == 'a':
            x = torch.cat([audio_x, ], dim=-1)
        if self.args.modality == 'v':
            x = torch.cat([video_x, ], dim=-1)
        if self.args.modality == 'tav':
            x = torch.cat([text_x, audio_x, video_x], dim=-1)
        if self.args.modality == 'ta':
            x = torch.cat([text_x, audio_x], dim=-1)
        if self.args.modality == 'tv':
            x = torch.cat([text_x, video_x], dim=-1)
        if self.args.modality == 'av':
            x = torch.cat([audio_x,video_x], dim=-1)

        x = self.norm(x)
        _, final_states = self.lstm(x)
        x = self.dropout(final_states[0][-1].squeeze(dim=0))
        x = F.relu(self.linear(x), inplace=True)
        x = self.dropout(x)
        output = self.out(x)
        res = {
            'M': output
        }
        return res 
        
class EF_CNN(nn.Module):
    """
    early fusion using cnn
    """
    def __init__(self, args):
        super(EF_CNN, self).__init__()

    def forward(self, text_x, audio_x, video_x):
        pass
