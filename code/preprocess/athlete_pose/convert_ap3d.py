"""Adapted from MotionBERT paper code (https://github.com/Walter0807/MotionBERT/blob/main/tools/convert_h36m.py)"""

import os
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from datareader_ap3d import DataReaderH36M
from tqdm import tqdm



def save_clips(subset_name, root_path, data, labels, vids, hmap_indices):
    len_clips = len(data)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Cartella dove si trovano i .pt (per creare il path relativo o assoluto)
    hmap_subfolder = 'valid' if subset_name == 'test' else 'train'

    for i in tqdm(range(len_clips)):
        data_dict = {
            "data_input": data[i],       # [81, 17, 3]
            "data_label": data[i],      # [81, 17, 3]
            "hmap_file": f"vid_{vids[i]}.pt", # Riferimento al file
            "hmap_relative_path": hmap_subfolder,
            "hmap_indices": hmap_indices[i]    # Gli 81 indici da caricare dal file .pt
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as fp:
            pickle.dump(data_dict, fp)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=81)
    parser.add_argument("--dataset-path", type=str, default='/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess')
    n_frames = parser.parse_args().n_frames
    dataset_path = parser.parse_args().dataset_path

    datareader = DataReaderH36M(n_frames=n_frames, sample_stride=1, data_stride_train=n_frames // 3,
                                 data_stride_test=n_frames, dt_root=dataset_path)
    train_bundle, test_bundle = datareader.get_sliced_data()

    root_path = f"{dataset_path}/frame_{n_frames}"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    save_clips("train", root_path, *train_bundle)
    save_clips("test", root_path, *test_bundle)


if __name__ == '__main__':
    main()