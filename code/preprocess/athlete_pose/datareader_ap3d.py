# Adapted from MotionBERT (https://github.com/Walter0807/MotionBERT/blob/main/lib/data/datareader_h36m.py)

import numpy as np
import random
import pickle
import pdb

random.seed(0)

def read_pkl(data_url):
    file = open(data_url, 'rb')
    content = pickle.load(file)
    file.close()
    return content

def resample(ori_len, target_len, replay=False, randomness=True):
    """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68"""
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len - target_len)
            return range(st, st + target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel * low + (1 - sel) * high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape) * interval + even
            result = np.clip(result, a_min=0, a_max=ori_len - 1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result

def split_clips(vid_list, n_frames, data_stride):
    """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L91"""
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i < len(vid_list):
        i += 1
        if i - st == n_frames:
            result.append(range(st, i))
            saved.add(vid_list[i - 1])
            st = st + data_stride
            n_clips += 1
        if i == len(vid_list):
            break
        if vid_list[i] != vid_list[i - 1]:
            if not (vid_list[i - 1] in saved):
                resampled = resample(i - st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i - 1])
            st = i
    return result

class DataReaderH36M(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root='data/motion3d'):
        self.gt_testset = None
        self.split_id_test = None
        self.test_hw = None
        test_file = 'test.pkl'
        #read the data
        self.dt_dataset = read_pkl('%s/%s' % (dt_root, test_file))
        self.dt_dataset = self.preprocess(self.dt_dataset)
        # self.dt_dataset = read_pkl('%s/%s' % (dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence

    def read_2d(self):
        testset = np.array(self.dt_dataset['joint_2d'], dtype=np.float32)[::self.sample_stride, :, :2]  # [N, 17, 2]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['camera_name']):
            res_w, res_h = self.dt_dataset['video_w'][idx], self.dt_dataset['video_h'][idx]
            testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        if self.read_confidence:
            if 'confidence' in self.dt_dataset.keys():
                test_confidence = self.dt_dataset['confidence'][::self.sample_stride].astype(np.float32)
                if len(test_confidence.shape) == 2:
                    test_confidence = test_confidence[:, :, None]
            else:
                # No conf provided, fill with 1.
                test_confidence = np.ones(testset.shape)[:, :, 0:1]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return testset

    def read_3d(self):
        test_labels = np.array(self.dt_dataset['joint3d_image'], dtype=np.float32)[::self.sample_stride, :, :3] # [N, 17, 3]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['camera_name']):
            res_w, res_h = self.dt_dataset['video_w'][idx], self.dt_dataset['video_h'][idx]
            test_labels[idx, :, :2] = test_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            test_labels[idx, :, 2:] = test_labels[idx, :, 2:] / res_w * 2

        return test_labels

    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            res_w, res_h = self.dt_dataset['train']['video_w'][idx], self.dt_dataset['train']['video_h'][idx]
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw

    def get_split_id(self):
        if self.split_id_test is not None:
            return self.split_id_test
        vid_list_test = self.dt_dataset['source'][::self.sample_stride]
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_test

    def turn_into_test_clips(self, data):
        """Converts (total_frames, ...) tensor to (n_clips, n_frames, ...) based on split_id_test"""
        split_id_test = self.get_split_id()
        data = data[split_id_test]
        return data

    def get_hw(self):
        test_hw = self.read_hw() 
        test_hw = self.turn_into_test_clips(test_hw)[:, 0, :]  
        return test_hw

    def get_sliced_data(self):
        test_data = self.read_2d() 
        test_labels = self.read_3d()  
        split_id_test = self.get_split_id()
        test_data = test_data[split_id_test]  
        test_labels = test_labels[split_id_test]  
        # pdb.set_trace()
        return test_data, test_labels

    def denormalize(self, test_data, all_sequence=False):
        if all_sequence:
            test_data = self.turn_into_test_clips(test_data)

        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])
        assert len(data) == len(test_hw), f"Data n_clips is {len(data)} while test_hw size is {len(test_hw)}"
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data  # [n_clips, -1, 17, 3]

    def preprocess(self, dataset):
        #       Preprocess dataset
        new_dataset = {'joint_2d':[], 'joint3d_image':[], 'camera_name':[], 'source':[],'video_w':[],'video_h':[],'joints_2.5d_image':[], '2.5d_factor':[],'action':[]}

        for i in range(len(dataset)):
            dict_i = dataset[i]
            new_dataset['joint_2d'].append(dict_i['joint_3d_image'])
            new_dataset['joint3d_image'].append(dict_i['joint_3d_image'])
            new_dataset['camera_name'].append(dict_i['cameraid'])
            new_dataset['source'].append(dict_i['videoid'])
            new_dataset['video_w'].append(dict_i['video_width'])
            new_dataset['video_h'].append(dict_i['video_height'])
            new_dataset['2.5d_factor'].append(1/dict_i['ratio'])
            new_dataset['joints_2.5d_image'].append(np.array(dict_i['joint_3d_image'])/ dict_i['ratio'])
            new_dataset['action'].append(dict_i['action'])
        return new_dataset
