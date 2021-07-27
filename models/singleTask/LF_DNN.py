"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from models.subNets.FeatureNets import SubNet, TextSubNet

__all__ = ['LF_DNN','LF_DNN1']


class LF_DNN1(nn.Module):
    """
    late fusion using DNN with three level residual network
    """

    def __init__(self, args):
        super(LF_DNN1, self).__init__()
        self.args = args
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_out = args.text_out
        self.post_fusion_dim1,self.post_fusion_dim2 = args.post_fusion_dim1,args.post_fusion_dim2
        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts

        if self.args.variant == 'v2':
            self.v2_get_insize()
        if self.args.variant == 'v0':
            self.v0_get_insize()
        if self.args.variant == 'v1':
            self.v1_get_insize()

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_prob)
        # define the post_fusion layers
        self.post_fusion_dropout1 = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_dropout2 = nn.Dropout(p=self.post_fusion_prob)

        self.post_fusion_layer_1 = nn.Linear(self.insize1, self.post_fusion_dim1)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim1, self.post_fusion_dim1)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim1, self.post_fusion_dim1)

        self.post_fusion_layer_4 = nn.Linear(self.insize2, self.post_fusion_dim2)
        self.post_fusion_layer_5 = nn.Linear(self.post_fusion_dim2, self.post_fusion_dim2)
        self.post_fusion_layer_6 = nn.Linear(self.post_fusion_dim2, self.post_fusion_dim2)

        self.post_fusion_layer_7 = nn.Linear(self.post_fusion_dim2, 1)

    def v1_get_insize(self):
        text_in, audio_in, video_in = self.args.feature_dims
        texth,audioh,videoh = self.args.hidden_dims
        pd1,pd2 = self.args.post_fusion_dim1,self.args.post_fusion_dim2

        if self.args.modality == 't':
            in_size1 = texth
        if self.args.modality == 'a':
            in_size1 = audioh
        if self.args.modality == 'v':
            in_size1 = videoh 
        if self.args.modality == 'tav':
            in_size1 = texth + videoh + audioh
        if self.args.modality == 'ta':
            in_size1 = texth + audioh
        if self.args.modality == 'tv':
            in_size1 = texth + videoh 
        if self.args.modality == 'av':
            in_size1 = videoh + audioh
        
        self.post_fusion_dim1 = pd2
        self.insize1 = in_size1
        self.insize2 = in_size1
        self.post_fusion_dim2 = in_size1

    def v0_get_insize(self):
        text_in, audio_in, video_in = self.args.feature_dims
        texth,audioh,videoh = self.args.hidden_dims
        pd1,pd2 = self.args.post_fusion_dim1,self.args.post_fusion_dim2

        if self.args.modality == 't':
            in_size1 = texth
        if self.args.modality == 'a':
            in_size1 = audioh
        if self.args.modality == 'v':
            in_size1 = videoh 
        if self.args.modality == 'tav':
            in_size1 = texth + videoh + audioh
        if self.args.modality == 'ta':
            in_size1 = texth + audioh
        if self.args.modality == 'tv':
            in_size1 = texth + videoh 
        if self.args.modality == 'av':
            in_size1 = videoh + audioh
        
        self.post_fusion_dim1 = pd2
        self.insize1 = in_size1
        self.insize2 = in_size1

    def v2_get_insize(self):
        text_in, audio_in, video_in = self.args.feature_dims
        texth,audioh,videoh = self.args.hidden_dims
        pd1,pd2 = self.args.post_fusion_dim1,self.args.post_fusion_dim2

        if self.args.modality == 't':
            in_size1 = text_in + texth
            in_size2 = pd1 + text_in 
        if self.args.modality == 'a':
            in_size1 = audio_in + audioh
            in_size2 = pd1 + audio_in 
        if self.args.modality == 'v':
            in_size1 = video_in + videoh 
            in_size2 = pd1 + video_in 
        if self.args.modality == 'tav':
            in_size1 = text_in + audio_in + video_in + texth + videoh + audioh
            in_size2 = pd1 + text_in + audio_in + video_in 
        if self.args.modality == 'ta':
            in_size1 = text_in + audio_in + texth + audioh
            in_size2 = pd1 + text_in + audio_in 
        if self.args.modality == 'tv':
            in_size1 = text_in + video_in + texth + videoh 
            in_size2 = pd1 + text_in + video_in 
        if self.args.modality == 'av':
            in_size1 = audio_in + video_in + videoh + audioh
            in_size2 = pd1 + audio_in + video_in 

        self.insize1 = in_size1
        self.insize2 = in_size2
        if self.args.speaker :
            # pdb.set_trace()
            self.insize2 = in_size2 + self.args.speakers


    def v1(self, text_x, audio_x, video_x, text_h, audio_h, video_h ):
        if self.args.modality == 't':
            x = torch.cat([text_h ], dim=-1)
        if self.args.modality == 'a':
            x = torch.cat([audio_h ], dim=-1)
        if self.args.modality == 'v':
            x = torch.cat([video_h ], dim=-1)
        if self.args.modality == 'tav':
            x = torch.cat([text_h, video_h, audio_h, ], dim=-1)
        if self.args.modality == 'ta':
            x = torch.cat([text_h, audio_h, ], dim=-1)
        if self.args.modality == 'tv':
            x = torch.cat([text_h, video_h, ], dim=-1)
        if self.args.modality == 'av':
            x = torch.cat([video_h, audio_h,], dim=-1)

        fusion_h = x
        x = self.post_fusion_dropout1(fusion_h)
        x = self.post_fusion_layer_7(x)

        output = x

        return output,fusion_h

    def v0(self, text_x, audio_x, video_x, text_h, audio_h, video_h ):
        if self.args.modality == 't':
            x = torch.cat([text_h ], dim=-1)
        if self.args.modality == 'a':
            x = torch.cat([audio_h ], dim=-1)
        if self.args.modality == 'v':
            x = torch.cat([video_h ], dim=-1)
        if self.args.modality == 'tav':
            x = torch.cat([text_h, video_h, audio_h, ], dim=-1)
        if self.args.modality == 'ta':
            x = torch.cat([text_h, audio_h, ], dim=-1)
        if self.args.modality == 'tv':
            x = torch.cat([text_h, video_h, ], dim=-1)
        if self.args.modality == 'av':
            x = torch.cat([video_h, audio_h,], dim=-1)

        fusion_h = x
        x = self.post_fusion_dropout1(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        x = self.post_fusion_layer_7(x)

        output = x

        return output,fusion_h

    def v2(self, text_x, audio_x, video_x, text_h, audio_h, video_h, speaker_x ):
        if self.args.modality == 't':
            x = torch.cat([text_x, text_h ], dim=-1)
        if self.args.modality == 'a':
            x = torch.cat([audio_x, audio_h ], dim=-1)
        if self.args.modality == 'v':
            x = torch.cat([video_x, video_h ], dim=-1)
        if self.args.modality == 'tav':
            x = torch.cat([text_h, video_h, audio_h, text_x, video_x, audio_x], dim=-1)
        if self.args.modality == 'ta':
            x = torch.cat([text_h, audio_h, text_x, audio_x], dim=-1)
        if self.args.modality == 'tv':
            x = torch.cat([text_h, video_h, text_x, video_x], dim=-1)
        if self.args.modality == 'av':
            x = torch.cat([video_h, audio_h,video_x, audio_x], dim=-1)

        fusion_h = x
        x = self.post_fusion_dropout1(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        x = F.relu(self.post_fusion_layer_3(x), inplace=True)

        if self.args.modality == 't':
            x = torch.cat([x, text_x, ], dim=-1)
        if self.args.modality == 'a':
            x = torch.cat([x, audio_x, ], dim=-1)
        if self.args.modality == 'v':
            x = torch.cat([x, video_x, ], dim=-1)
        if self.args.modality == 'tav':
            x = torch.cat([x, text_x, video_x, audio_x], dim=-1)
        if self.args.modality == 'ta':
            x = torch.cat([x, text_x, audio_x], dim=-1)
        if self.args.modality == 'tv':
            x = torch.cat([x, text_x, video_x], dim=-1)
        if self.args.modality == 'av':
            x = torch.cat([x, video_x, audio_x], dim=-1)

        x = self.post_fusion_dropout2(x)
        if self.args.speaker :
            x = torch.cat([x, speaker_x], dim=-1)
        x = F.relu(self.post_fusion_layer_4(x), inplace=True)
        x = F.relu(self.post_fusion_layer_5(x), inplace=True)
        x = F.relu(self.post_fusion_layer_6(x), inplace=True)
        output = self.post_fusion_layer_7(x)

        return output,fusion_h


    def forward(self, text_x, audio_x, video_x, speaker_x = None ):
        audio_x = audio_x.squeeze(1)
        video_x = video_x.squeeze(1)
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        # for use bert embedding,we use [CLS] token as the repr of the whole sentence. or
        # 使用平均值 ，但是应该考虑 填充的影响。

        # pdb.set_trace()
        if self.args.context :
            ctext_x = text_x[:, 0, :]
            text_x = text_x[:, self.args.csindex, :]
            text_x = (ctext_x + text_x) / 2 
        else:
            text_x = text_x[:, 0, :]
        text_h = self.text_subnet(text_x)

        if self.args.variant == 'v2':
            output,fusion_h = self.v2(text_x, audio_x, video_x, text_h, audio_h, video_h, speaker_x )
        if self.args.variant == 'v0':
            output,fusion_h = self.v0(text_x, audio_x, video_x, text_h, audio_h, video_h )
        if self.args.variant == 'v1':
            output,fusion_h = self.v1(text_x, audio_x, video_x, text_h, audio_h, video_h )

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
            'M': output
        }

        return res


class LF_DNN(nn.Module):
    """
    late fusion using DNN
    """

    def __init__(self, args):
        super(LF_DNN, self).__init__()

        # 'feature_dims': (768, 33, 709)
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        # 'hidden_dims': (128, 16, 128)
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        # 'text_out': 32
        self.text_out = args.text_out
        # 'post_fusion_dim': 32
        self.post_fusion_dim = args.post_fusion_dim

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts
        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)
        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_hidden + self.audio_hidden,
                                             self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

    def forward(self, text_x, audio_x, video_x):
        audio_x = audio_x.squeeze(1)
        video_x = video_x.squeeze(1)
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)

        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)

        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        output = self.post_fusion_layer_3(x)
        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
            'M': output
        }

        return res
