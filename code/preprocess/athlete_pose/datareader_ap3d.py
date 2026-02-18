# Adapted from MotionBERT (https://github.com/Walter0807/MotionBERT/blob/main/lib/data/datareader_h36m.py)

import numpy as np
import random
import pickle
import pdb
import torch
import os
from itertools import groupby

random.seed(42)

def extract_keypoints_torch(heatmaps, metas_tensor):
    """
    Versione ottimizzata per il caricamento massivo
    heatmaps: [T, 17, 64, 64]
    metas_tensor: [T, 5] (i, x1, y1, h, w)
    """
    T, J, Hh, Wh = heatmaps.shape 
    heatmaps_flat = heatmaps.view(T, J, -1)
    conf_max, idx_max = torch.max(heatmaps_flat, dim=2)

    y_local = (torch.div(idx_max, Wh, rounding_mode='floor')).float() + 0.5
    x_local = (idx_max % Wh).float() + 0.5

    x1, y1 = metas_tensor[:, 1], metas_tensor[:, 2]
    crop_h, crop_w = metas_tensor[:, 3], metas_tensor[:, 4]

    scale_x = (crop_w / Wh)[:, None]
    scale_y = (crop_h / Hh)[:, None]
    
    x_global = x_local * scale_x + x1[:, None]
    y_global = y_local * scale_y + y1[:, None]

    # [T, 17, 3] -> (x, y, conf)
    return torch.stack([x_global, y_global, conf_max], dim=2)

def coco_to_h36m(keypoints_out):
    """
    Mapping COCO (17 joints) -> H36M (17 joints)
    Input: [T, 17, 3]
    """
    T = keypoints_out.shape[0]
    h36m_kps = torch.zeros((T, 17, 3), device=keypoints_out.device)

    # 0: Pelvis (Centro tra le anche)
    h36m_kps[:, 0, :] = (keypoints_out[:, 11, :] + keypoints_out[:, 12, :]) * 0.5
    # 1: Right Hip / 2: Right Knee / 3: Right Ankle
    h36m_kps[:, 1, :] = keypoints_out[:, 12, :]
    h36m_kps[:, 2, :] = keypoints_out[:, 14, :]
    h36m_kps[:, 3, :] = keypoints_out[:, 16, :]
    # 4: Left Hip / 5: Left Knee / 6: Left Ankle
    h36m_kps[:, 4, :] = keypoints_out[:, 11, :]
    h36m_kps[:, 5, :] = keypoints_out[:, 13, :]
    h36m_kps[:, 6, :] = keypoints_out[:, 15, :]
    # 8: Thorax (Centro spalle)
    h36m_kps[:, 8, :] = (keypoints_out[:, 5, :] + keypoints_out[:, 6, :]) * 0.5
    # 7: Spine (Centro tra Pelvis e Thorax)
    h36m_kps[:, 7, :] = (h36m_kps[:, 0, :] + h36m_kps[:, 8, :]) * 0.5
    # 9: Neck (Naso come proxy) / 10: Head (Centro orecchie)
    h36m_kps[:, 9, :] = keypoints_out[:, 0, :]
    h36m_kps[:, 10, :] = (keypoints_out[:, 1, :] + keypoints_out[:, 2, :]) * 0.5
    # 11, 12, 13: Left Shoulder, Elbow, Wrist
    h36m_kps[:, 11, :] = keypoints_out[:, 5, :]
    h36m_kps[:, 12, :] = keypoints_out[:, 7, :]
    h36m_kps[:, 13, :] = keypoints_out[:, 9, :]
    # 14, 15, 16: Right Shoulder, Elbow, Wrist
    h36m_kps[:, 14, :] = keypoints_out[:, 6, :]
    h36m_kps[:, 15, :] = keypoints_out[:, 8, :]
    h36m_kps[:, 16, :] = keypoints_out[:, 10, :]

    return h36m_kps

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
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None

        #read the data
        dt_train = read_pkl(os.path.join(dt_root, 'train.pkl'))
        dt_valid = read_pkl(os.path.join(dt_root, 'valid.pkl'))

        self.dt_root = dt_root
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence

        # 1. Preprocess: Estrae i punti e sistema i percorsi heatmap
        # Passiamo i dizionari originali alla funzione preprocess
        raw_dataset = {'train': dt_train, 'test': dt_valid}
        self.dt_dataset = self.preprocess(raw_dataset)


    def _empty_info(self, test=False):
        info = {'joint_2d':[], 'joint3d_image':[], 'camera_name':[], 'source':[], 'video_w':[], 'video_h':[], 'confidence':[]}
        if test:
            info.update({'joints_2.5d_image':[], '2.5d_factor':[], 'action':[]})
        return info

    def read_2d(self):
        """ Ritorna le coordinate 2D e confidenza con stride applicato """
        # Estraiamo e applichiamo lo stride a tutto il blocco
        train_kps = np.array(self.dt_dataset['train']['joint_2d'], dtype=np.float32)
        test_kps = np.array(self.dt_dataset['test']['joint_2d'], dtype=np.float32)
        
        # Normalizzazione spaziale
        for i in range(len(train_kps)):
            w, h = self.dt_dataset['train']['video_w'][i], self.dt_dataset['train']['video_h'][i]
            train_kps[i] = train_kps[i] / w * 2 - [1, h / w]
            
        for i in range(len(test_kps)):
            w, h = self.dt_dataset['test']['video_w'][i], self.dt_dataset['test']['video_h'][i]
            test_kps[i] = test_kps[i] / w * 2 - [1, h / w]

        train_conf = np.array(self.dt_dataset['train']['confidence'], dtype=np.float32)[:, :, None]
        test_conf = np.array(self.dt_dataset['test']['confidence'], dtype=np.float32)[:, :, None]

        trainset = np.concatenate((train_kps, train_conf), axis=2)
        testset = np.concatenate((test_kps, test_conf), axis=2)

        return trainset[::self.sample_stride], testset[::self.sample_stride]

    def read_3d(self):
        """ Ritorna le coordinate 3D (Labels) con stride applicato """
        train_labels = np.array(self.dt_dataset['train']['joint3d_image'], dtype=np.float32)
        test_labels = np.array(self.dt_dataset['test']['joint3d_image'], dtype=np.float32)

        # Normalizzazione (stessa logica del 2D)
        for i in range(len(train_labels)):
            w, h = self.dt_dataset['train']['video_w'][i], self.dt_dataset['train']['video_h'][i]
            train_labels[i, :, :2] = train_labels[i, :, :2] / w * 2 - [1, h / w]
            train_labels[i, :, 2:] = train_labels[i, :, 2:] / w * 2

        return train_labels[::self.sample_stride], test_labels[::self.sample_stride]

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
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]  # (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]  # (566920,)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test

    def turn_into_test_clips(self, data):
        """Converts (total_frames, ...) tensor to (n_clips, n_frames, ...) based on split_id_test"""
        split_id_train, split_id_test = self.get_split_id()
        data = data[split_id_test]
        return data

    def get_hw(self):
        test_hw = self.read_hw() 
        test_hw = self.turn_into_test_clips(test_hw)[:, 0, :]  
        return test_hw

    # Aggiungi questo metodo alla classe DataReaderH36M
    def get_sliced_data(self):
        train_data, test_data = self.read_2d() 
        train_labels, test_labels = self.read_3d()  
        split_id_train, split_id_test = self.get_split_id()
        
        # Estrarre i metadati per sapere da dove vengono i frame
        # 'source' contiene il videoid, 'camera_name' contiene l'indice frame originale
        train_source = np.array(self.dt_dataset['train']['source'])[::self.sample_stride]
        train_frame_idx = np.array(self.dt_dataset['train']['camera_name'])[::self.sample_stride]
        
        test_source = np.array(self.dt_dataset['test']['source'])[::self.sample_stride]
        test_frame_idx = np.array(self.dt_dataset['test']['camera_name'])[::self.sample_stride]
    
        # Affettiamo tutto!
        train_data = train_data[split_id_train]
        train_labels = train_labels[split_id_train]
        # Prendiamo il videoid del primo frame della clip (assumendo che la clip non crossi video)
        train_vids = train_source[split_id_train][:, 0] 
        train_hmap_indices = train_frame_idx[split_id_train]
    
        test_data = test_data[split_id_test]
        test_labels = test_labels[split_id_test]
        test_vids = test_source[split_id_test][:, 0]
        test_hmap_indices = test_frame_idx[split_id_test]
    
        return (train_data, train_labels, train_vids, train_hmap_indices), \
               (test_data, test_labels, test_vids, test_hmap_indices)

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
        new_dataset = {'train': self._empty_info(), 'test': self._empty_info(test=True)}
        
        for split in ['train', 'test']:
            # FIX: Mappatura corretta delle cartelle su disco
            folder_on_disk = 'valid' if split == 'test' else 'train'
            print(f"🔄 Pre-processing {split} (cartella sorgente: {folder_on_disk})...")
            
            # Raggruppiamo per videoid per processare i file .pt uno alla volta
            data_by_video = groupby(dataset[split], key=lambda x: x['videoid'])
            
            for vid_id, frames in data_by_video:
                frame_list = list(frames)
                hmap_file = os.path.join(self.dt_root, 'heatmap', folder_on_disk, f"vid_{vid_id}.pt")
                
                if not os.path.exists(hmap_file):
                    print(f"⚠️ Saltato vid_{vid_id}: file non trovato.")
                    continue

                # Carichiamo le heatmap per estrarre i punti 2D precisi
                heatmaps = torch.load(hmap_file, map_location='cpu')
                T_hmap = heatmaps.shape[0]
                T_meta = len(frame_list)
                T = min(T_hmap, T_meta) # Sicurezza contro disallineamenti frame
                
                # Estrazione coordinate 2D dalle heatmap (usando la tua funzione ottimizzata)
                metas = []
                for i in range(T):
                    box = frame_list[i]['box']
                    metas.append([i, box[0], box[1], box[3]-box[1], box[2]-box[0]])
                
                metas_tensor = torch.tensor(metas)
                kps_coco = extract_keypoints_torch(heatmaps[:T], metas_tensor)
                kps_h36m = coco_to_h36m(kps_coco).numpy() # [T, 17, 3]

                # Salvataggio nel nuovo dataset
                for i in range(T):
                    new_dataset[split]['joint_2d'].append(kps_h36m[i, :, :2])
                    new_dataset[split]['confidence'].append(kps_h36m[i, :, 2])
                    new_dataset[split]['joint3d_image'].append(frame_list[i]['joint_3d_image'])
                    new_dataset[split]['source'].append(frame_list[i]['videoid'])
                    # IMPORTANTE: salviamo l'indice del frame originale per lo stride futuro
                    new_dataset[split]['camera_name'].append(i) # Qui i è l'indice nel file .pt
                    new_dataset[split]['video_w'].append(frame_list[i]['video_width'])
                    new_dataset[split]['video_h'].append(frame_list[i]['video_height'])
                    
                    if split == 'test':
                        new_dataset[split]['action'].append(frame_list[i]['action'])

        return new_dataset

