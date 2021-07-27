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


class getFeatures():
    def __init__(self, working_dir, openface2Path, pretrainedBertPath):
        self.data_dir = os.path.join(working_dir, 'Processed')
        self.label_path = os.path.join(working_dir, 'metadata/sentiment')
        # padding
        self.padding_mode = 'zeros'
        self.padding_location = 'back'
        # toolkits path
        self.openface2Path = openface2Path
        self.pretrainedBertPath = pretrainedBertPath

        tokenizer_class = BertTokenizer
        model_class = BertModel
        pretrained_weights = self.pretrainedBertPath
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)


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
        # tokenizer_class = BertTokenizer
        # model_class = BertModel
        # pretrained_weights = self.pretrainedBertPath
        #
        # tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        # model = model_class.from_pretrained(pretrained_weights)
        # add_special_tokens will add start and end token


        input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])

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
        final_length = int(np.mean(lens) + 3 * np.std(lens))
        # padding sequences to final_length
        final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        for i, s in enumerate(sequences):
            final_sequence[i] = self.__padding(s, final_length)

        return final_sequence

    def handleImages(self):
        image_dirs = sorted(glob(os.path.join(self.data_dir, 'video/AlignedFaces', '*/*')))
        # image_dirs = sorted(glob(os.path.join(self.data_dir, 'video/AlignedFaces', 'video_0008/0008')))
        # df_label_T = pd.read_csv(os.path.join(self.label_path, 'label_T_dn.csv'))

        # for image_dir in tqdm(image_dirs[:len(df_label_T)]):
        # for image_dir in tqdm(image_dirs[1449:]):
        for image_dir in tqdm(image_dirs):
            output_dir = image_dir.replace('AlignedFaces', 'OpenFace2')

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cmd = self.openface2Path + ' -fdir ' + image_dir + ' -out_dir ' + output_dir
            os.system(cmd)


    def get_appropriate_dataset(self,train_data):
        features_T, features_A, features_V = [], [], []
        label_M = []
        segments = []

        for example in tqdm(train_data):
            (words, visual, acoustic), label_id, segment = example
            # 处理文本时，应该注意中文和英文的区别。
            embedding_T = self.__getTextEmbedding(words)
            embedding_T = embedding_T[1:-1]
            embedding_V = visual
            embedding_A = acoustic
            features_T.append(embedding_T)
            features_A.append(embedding_A)
            features_V.append(embedding_V)
            label_M.append(label_id)
            segments.append(segment)

        # padding
        feature_T = self.__paddingSequence(features_T)
        feature_A = self.__paddingSequence(features_A)
        feature_V = self.__paddingSequence(features_V)


        train_pkl = {'vision': feature_V, 'text': feature_T, 'audio': feature_A, 'labels': label_M,
                     'segments': segments}
        return train_pkl

    def results(self, output_dir):

        with open(os.path.join(self.data_dir,"mustard_unaligned.pkl"), "rb") as handle:
            data = pickle.load(handle)

        train_data = data["train"]
        dev_data = data["dev"]
        test_data = data["test"]

        train_pkl = self.get_appropriate_dataset(train_data)
        dev_pkl = self.get_appropriate_dataset(dev_data)
        test_pkl = self.get_appropriate_dataset(test_data)

        result_pkl = {"train": train_pkl, "valid": dev_pkl, "test": test_pkl}

        output_dir_my = os.path.join(self.data_dir, output_dir)

        if not os.path.exists(output_dir_my):
            os.makedirs(output_dir_my)
        # save
        save_path = os.path.join(self.data_dir, output_dir, 'mustard.pkl')
        DATASET_FILE = save_path
        if not os.path.exists(DATASET_FILE):
            pickle.dump(result_pkl, open(DATASET_FILE, 'wb'), protocol=2)
        print('Features are saved in %s!' % save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/datasets/MOSI',
                        help='path to MOSI')
    parser.add_argument('--openface2Path', type=str, default='D:\\tools\\openface2\\FeatureExtraction.exe',
                        help='path to FeatureExtraction tool in openface2')
    parser.add_argument('--pretrainedBertPath', type=str, default='/tools/chinese_bert/',
                        help='path to pretrained bert directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # data_dir = '/path/to/MSA-ZH'
    gf = getFeatures(args.data_dir, args.openface2Path, args.pretrainedBertPath)

    # gf.handleImages()

    gf.results('features')
