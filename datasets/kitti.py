import sys
import os
from typing import Mapping
import numpy as np
import torch.utils.data as data

__all__ = ['KITTI']

class KITTI(data.Dataset):
    def __init__(self, train, transform, num_points, data_dir, remove_ground=True):
        """
        Args:
            train (bool): true--> train dataset, false--> test dataset
            transform (callable)
        """
        super(KITTI, self).__init__()
        self.data_dir = os.path.join(data_dir, 'KITTI_processed_occ_final')
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.samples_path = self.build_dataset()

        if len(self.samples_path) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + self.data_dir + "\n"))
        

    def __len__(self) -> int:
        return len(self.samples_path)


    def __getitem__(self, index):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples_path[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)
        
        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

    
    def build_dataset_path(self):
        do_mappint = True
        base_path = os.path.realpath(os.path.expanduser(self.data_dir))

        all_paths = os.walk(base_path)
        used_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])
        try:
            assert (len(used_paths) == 200)
        except AssertionError:
            print('len(useful_paths) == 200 failed.', len(used_paths))
        
        if do_mappint:
            mapping_path = os.path.join(os.path.dirname(__file__), "KITTI_mappint.txt")
            print(f'mapping_path: {mapping_path}')

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            used_paths = [path for path in used_paths if lines[int(os.path.split(path)[-1])] != '']

        res_paths = used_paths
        return res_paths
    
    
    def pc_loader(self, path):
        pc1 = np.load(os.path.join(path, 'pc1.npy'))
        pc2 = np.load(os.path.join(path, 'pc2.npy'))

        if self.remove_ground:
            is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            not_ground = np.logical_not(is_ground)

            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]