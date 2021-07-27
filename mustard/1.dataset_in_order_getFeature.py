import os
import argparse
import librosa
import struct
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import pickle

import torch
from transformers import *
from globals import M_FUSION_DATA_PATH ,raw_data_path,tool_path


class getFeatures():
    def __init__(self, working_dir,  pretrainedBertPath):
        self.data_dir = working_dir
        self.padding_mode = 'zeros'
        self.padding_location = 'back'
        self.pretrainedBertPath = pretrainedBertPath
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainedBertPath)
        self.model = BertModel.from_pretrained(self.pretrainedBertPath)


    def __read_hog(self, filename, batch_size=5000):
        """
        From: https://gist.github.com/btlorch/6d259bfe6b753a7a88490c0607f07ff8
        Read HoG features file created by OpenFace.
        For each frame, OpenFace extracts 12 * 12 * 31 HoG features, i.e., num_features = 4464. These features are stored in row-major order.
        :param filename: path to .hog file created by OpenFace
        :param batch_size: how many rows to read at a time
        :return: is_valid, hog_features
            is_valid: ndarray of shape [num_frames]
            hog_features: ndarray of shape [num_frames, num_features]
        """
        all_feature_vectors = []
        with open(filename, "rb") as f:
            num_cols, = struct.unpack("i", f.read(4))
            num_rows, = struct.unpack("i", f.read(4))
            num_channels, = struct.unpack("i", f.read(4))

            # The first four bytes encode a boolean value whether the frame is valid
            num_features = 1 + num_rows * num_cols * num_channels
            feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
            feature_vector = np.array(feature_vector).reshape((1, num_features))
            all_feature_vectors.append(feature_vector)

            # Every frame contains a header of four float values: num_cols, num_rows, num_channels, is_valid
            num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
            # Read in batches of given batch_size
            num_floats_to_read = num_floats_per_feature_vector * batch_size
            # Multiply by 4 because of float32
            num_bytes_to_read = num_floats_to_read * 4

            while True:
                bytes = f.read(num_bytes_to_read)
                # For comparison how many bytes were actually read
                num_bytes_read = len(bytes)
                assert num_bytes_read % 4 == 0, "Number of bytes read does not match with float size"
                num_floats_read = num_bytes_read // 4
                assert num_floats_read % num_floats_per_feature_vector == 0, "Number of bytes read does not match with feature vector size"
                num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector

                feature_vectors = struct.unpack("{}f".format(num_floats_read), bytes)
                # Convert to array
                feature_vectors = np.array(feature_vectors).reshape(
                    (num_feature_vectors_read, num_floats_per_feature_vector))
                # Discard the first three values in each row (num_cols, num_rows, num_channels)
                feature_vectors = feature_vectors[:, 3:]
                # Append to list of all feature vectors that have been read so far
                all_feature_vectors.append(feature_vectors)

                if num_bytes_read < num_bytes_to_read:
                    break

            # Concatenate batches
            all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)

            # Split into is-valid and feature vectors
            is_valid = all_feature_vectors[:, 0]
            feature_vectors = all_feature_vectors[:, 1:]

            return is_valid, feature_vectors

    def __getTextEmbedding(self, text):
        # 是否可以批量处理发挥GPU的性能。
        input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])
        # [0] 表示的是 一次处理1个 batchsize=1
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples

        return last_hidden_states.squeeze().numpy()

    def __getAudioEmbedding(self, audio_path):
        y, sr = librosa.load(audio_path)
        # using librosa to get audio features (f0, mfcc, cqt)
        hop_length = 512  # hop_length smaller, seq_len larger
        f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T  # (seq_len, 1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T  # (seq_len, 20)
        cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T  # (seq_len, 12)

        return np.concatenate([f0, mfcc, cqt], axis=-1)  # (seq_len, 33)

    def __getVideoEmbedding(self, csv_path, pool_size=5):
        df = pd.read_csv(csv_path)

        features, local_features = [], []
        for i in range(len(df)):
            local_features.append(np.array(df.loc[i][df.columns[5:]]))
            if (i + 1) % pool_size == 0:
                features.append(np.array(local_features).mean(axis=0))
                local_features = []
        if len(local_features) != 0:
            features.append(np.array(local_features).mean(axis=0))
        return np.array(features)

    def __padding(self, feature, MAX_LEN):
        """
        mode:
            zero: padding with 0
            normal: padding with normal distribution
        location: front / back
        """
        assert self.padding_mode in ['zeros', 'normal']
        assert self.padding_location in ['front', 'back']

        length = feature.shape[0]
        if length >= MAX_LEN:
            return feature[:MAX_LEN, :]

        if self.padding_mode == "zeros":
            pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
        elif self.padding_mode == "normal":
            mean, std = feature.mean(), feature.std()
            pad = np.random.normal(mean, std, (MAX_LEN - length, feature.shape[1]))

        feature = np.concatenate([pad, feature], axis=0) if (self.padding_location == "front") else \
            np.concatenate((feature, pad), axis=0)
        return feature

    def __paddingSequence(self, sequences):
        feature_dim = sequences[0].shape[-1]
        lens = [s.shape[0] for s in sequences]
        # confirm length using (mean + std)
        # 因为Mustard数据集 长度相差很大，因此取均值+1 std填充。
        print('mean , 1 * std : ',np.mean(lens) , 1 * np.std(lens))
        final_length = int(np.mean(lens) + 1 * np.std(lens))
        # padding sequences to final_length
        final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        for i, s in enumerate(sequences):
            final_sequence[i] = self.__padding(s, final_length)

        return final_sequence




    def get_appropriate_dataset(self,data):
        features_t_final, features_v_final, features_a_final  = [], [], []
        features_t_context, features_v_context, features_a_context  = [], [], []
        speaker_finals = []
        speaker_contexts = []
        labels = []
        segments = []

        #for example in data[:9]:
        for example in tqdm(data):
            (text_final,  video_final, audio_final,), text_context , video_context, audio_context, speaker_final, speaker_context, label, segment = example
            # 处理文本时，应该注意中文和英文的区别。
            embedding_t_final = self.__getTextEmbedding(text_final)
            # 保留CLS的特征用于整个句子的特征，在使用模态间处理的时候，需要去除首位词语特征。
            # embedding_T = embedding_T[1:-1]
            embedding_v_final = video_final
            embedding_a_final = audio_final

            features_t_final.append(embedding_t_final)
            features_v_final.append(embedding_v_final)
            features_a_final.append(embedding_a_final)

            embedding_t_context = self.__getTextEmbedding(text_context)
            embedding_v_context = video_context
            embedding_a_context = audio_context

            features_t_context.append(embedding_t_context)
            features_v_context.append(embedding_v_context)
            features_a_context.append(embedding_a_context)

            speaker_finals.append(speaker_final)
            speaker_contexts.append(speaker_context)
            labels.append(label)
            segments.append(segment)

        # padding
        feature_t_final = self.__paddingSequence(features_t_final)
        feature_v_final = self.__paddingSequence(features_v_final)
        feature_a_final = self.__paddingSequence(features_a_final)
        
        feature_t_context = self.__paddingSequence(features_t_context)
        feature_v_context = self.__paddingSequence(features_v_context)
        feature_a_context = self.__paddingSequence(features_a_context)

        pkl = {
                'text_final':feature_t_final, 
                'video_final':feature_v_final, 
                'audio_final':feature_a_final, 
                'text_context':feature_t_context, 
                'video_context':feature_v_context, 
                'audio_context':feature_a_context, 
                'speaker_final':speaker_finals,
                'speaker_context':speaker_contexts,
                'labels':labels,
                'segments':segments
                }

        return pkl

    def results(self, output_dir):
        with open(os.path.join(self.data_dir,"mustard.pkl"), "rb") as handle:
            data = pickle.load(handle)

        result_pkl = self.get_appropriate_dataset(data)

        if not os.path.exists(os.path.join(self.data_dir, output_dir)):
            os.makedirs(os.path.join(self.data_dir, output_dir))
        # save
        DATASET_FILE = os.path.join(self.data_dir, output_dir, 'mustard.pkl')
        if not os.path.exists(DATASET_FILE):
            pickle.dump(result_pkl, open(DATASET_FILE, 'wb'), protocol=2)
        print('Features are saved in %s' % save_path)

    def results33context(self,output_dir):
        with open(os.path.join(self.data_dir,"mustard_33_context.pkl"), "rb") as handle:
            data = pickle.load(handle)

        result_pkl = self.get_appropriate_dataset(data)

        if not os.path.exists(os.path.join(self.data_dir, output_dir)):
            os.makedirs(os.path.join(self.data_dir, output_dir))
        # save
        DATASET_FILE = os.path.join(self.data_dir, output_dir, 'mustard_33_context.pkl')
        #if not os.path.exists(DATASET_FILE):
        pickle.dump(result_pkl, open(DATASET_FILE, 'wb'), protocol=2)
        print('mustard_33_context are saved in %s' % DATASET_FILE)

    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=M_FUSION_DATA_PATH,
                        help='path to MOSI')
    # attention! for mustard is english,so select right pretrained models
    parser.add_argument('--pretrainedBertPath', type=str, default=tool_path+'bert-base-uncased',
                        help='path to pretrained bert directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gf = getFeatures(args.data_dir,  args.pretrainedBertPath)
    #gf.results('bert_mustard')
    gf.results33context('bert_mustard')
