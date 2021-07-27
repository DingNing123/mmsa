import os
import argparse
import librosa
import struct
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from transformers import *


class getFeatures():
    def __init__(self, working_dir, openface2Path, pretrainedBertPath):
        self.data_dir = os.path.join(working_dir, 'Processed')
        self.label_path = os.path.join(working_dir, 'metadata/sentiment')
        self.index_path = os.path.join(working_dir, 'metadata')

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
        hop_length = 5120  # hop_length smaller, seq_len larger
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

        for image_dir in tqdm(image_dirs):
            output_dir = image_dir.replace('AlignedFaces', 'OpenFace2')

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cmd = self.openface2Path + ' -fdir ' + image_dir + ' -out_dir ' + output_dir
            os.system(cmd)

    def __align(self, words, visual, acoustic):
        len_words = len(words)
        len_visual = visual.shape[0]
        len_acoustic = acoustic.shape[0]

        # 如果visual is too short ,extent it 太短了，需要扩展长度
        if len_words // len_visual >= 2:
            extend_times = len_words // len_visual
            result = np.zeros((extend_times*len_visual,visual.shape[-1]))
            for i in range(len_visual):
                result[i:(i+1)*extend_times] = visual[i]
            visual = result

        len_visual = visual.shape[0]

        # visual align
        if len_visual % len_words != 0:
            pad = len_words - len_visual % len_words
            left_pad = pad // 2
            right_pad = pad - left_pad
            left = visual[:left_pad]
            right = visual[-right_pad:]
            visual = np.concatenate([left,visual,right],axis=0)

        len_visual = visual.shape[0]
        assert len_visual % len_words == 0
        # pool
        visual = np.mean(visual.reshape(-1,len_words,visual.shape[-1]),axis=0)

        # 如果 the same : acoustic is too short ,extent it 太短了，需要扩展长度
        if len_words // len_acoustic >= 2:
            extend_times = len_words // len_acoustic
            result = np.zeros((extend_times * len_acoustic, acoustic.shape[-1]))
            for i in range(len_acoustic):
                result[i:(i + 1) * extend_times] = acoustic[i]
            acoustic = result

        len_acoustic = acoustic.shape[0]

        # acoustic align
        if len_acoustic % len_words != 0:
            pad = len_words - len_acoustic % len_words
            left_pad = pad // 2
            right_pad = pad - left_pad
            left = acoustic[:left_pad]
            right = acoustic[-right_pad:]
            acoustic = np.concatenate([left, acoustic, right], axis=0)

        len_acoustic = acoustic.shape[0]
        assert len_acoustic % len_words == 0
        # pool
        acoustic = np.mean(acoustic.reshape(-1, len_words, acoustic.shape[-1]), axis=0)

        return words, visual, acoustic

    def results(self, output_dir):
        df_label_T = pd.read_csv(os.path.join(self.label_path, 'label_T.csv'))
        df_label_A = pd.read_csv(os.path.join(self.label_path, 'label_A.csv'))
        df_label_V = pd.read_csv(os.path.join(self.label_path, 'label_V.csv'))
        df_label_M = pd.read_csv(os.path.join(self.label_path, 'label_M.csv'))

        data_points = []

        # for i in tqdm(range(10)):

        for i in tqdm(range(len(df_label_T))[:]):
            video_id, clip_id = df_label_T.loc[i, ['video_id', 'clip_id']]
            clip_id = '%04d' % clip_id
            words = df_label_T.loc[i, 'text']

            audio_path = os.path.join(self.data_dir, 'audio', video_id, clip_id + '.wav')
            acoustic = self.__getAudioEmbedding(audio_path)

            csv_path = os.path.join(self.data_dir, 'video/OpenFace2', video_id, clip_id, clip_id + '.csv')
            visual = self.__getVideoEmbedding(csv_path, pool_size=1)

            label_id = df_label_M.loc[i, 'label']
            label_id_t = df_label_T.loc[i, 'label']
            label_id_a = df_label_A.loc[i, 'label']
            label_id_v = df_label_V.loc[i, 'label']
            segment = video_id

            words, visual, acoustic = self.__align(words, visual, acoustic)
            data_point = (words, visual, acoustic), label_id, segment, (label_id_t,label_id_a,label_id_v)
            data_points.append(data_point)

        test_index = np.array(pd.read_csv(os.path.join(self.index_path, 'test_index.csv'))).reshape(-1)
        train_index = np.array(pd.read_csv(os.path.join(self.index_path, 'train_index.csv'))).reshape(-1)
        val_index = np.array(pd.read_csv(os.path.join(self.index_path, 'val_index.csv'))).reshape(-1)

        data_points = np.array(data_points)
        # for train dev test , need index
        train_pkl = data_points[train_index.astype(int)].tolist()
        dev_pkl = data_points[val_index.astype(int)].tolist()
        test_pkl = data_points[test_index.astype(int)].tolist()

        result_pkl = {"train": train_pkl, "dev": dev_pkl, "test": test_pkl}

        output_dir_my = os.path.join(self.data_dir, output_dir)

        if not os.path.exists(output_dir_my):
            os.makedirs(output_dir_my)

        import  pickle
        save_path = os.path.join(self.data_dir, output_dir, 'mmsa_mag.pkl')
        DATASET_FILE = save_path
        if not os.path.exists(DATASET_FILE):
            pickle.dump(result_pkl, open(DATASET_FILE, 'wb'), protocol=2)
        print('Features are saved in %s!' % save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/media/dn/85E803050C839C68/datasets/CH-SIMS',
                        help='path to CH-SIMS')
    parser.add_argument('--openface2Path', type=str,
                        default='/home/dn/tools/OpenFace-master/build/bin/FeatureExtraction',
                        help='path to FeatureExtraction tool in openface2')
    parser.add_argument('--pretrainedBertPath', type=str, default='/media/dn/85E803050C839C68/tools/chinese_bert/',
                        help='path to pretrained bert directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    gf = getFeatures(args.data_dir, args.openface2Path, args.pretrainedBertPath)

    # gf.handleImages()

    gf.results('features')
