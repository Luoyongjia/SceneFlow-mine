import re
import sys
import os
import numpy as np
import pptk
import torch.utils.data as data

__all__ = ["FlyingThings3DSubset"]

class FlyingThings3DSubset(data.Dataset):
    def __init__(self, train, transform, num_points, data_dir, full=True):
        """
        Args:
            train (bool): true--> train dataset, false--> test dataset
            transform (callable)
        """
        super(FlyingThings3DSubset).__init__()
        self.data_dir = os.path.join(data_dir, 'FlyingThings3D_subset_processed_35m')
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.samples_path = self.build_dataset(full)

        if len(self.samples_path) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + self.data_dir + "\n"))
        
    
    def __len__(self):
        return len(self.samples_path)
    

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples_path[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print(f'path {self.samples_path[index]} get pc1 is None.')
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)
        
        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples_path[index]


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    


    def build_dataset_path(self, full):
        """get samples' path list

        Args:
            full (bool): wether use the full dataset
        """

        base_path = os.path.realpath(os.path.expanduser(self.data_dir))
        base_path = os.path.join(dir, 'train') if self.train else os.path.join(dir, 'val')

        all_paths = os.walk(base_path)
        used_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])

        try:
            if self.train:
                assert (len(used_paths) == 19640)
            else:
                assert(len(used_paths) == 3824)
        except AssertionError:
            print('len(useful_paths) assert error', len(used_paths))
            sys.exit(1)
        
        if not full:
            res_paths = used_paths[::4]
        else:
            res_paths = used_paths
        
        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path: path to a dir, e.g., ./35mm_focallength/scene_forwards/slow/0791
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(os.path.join(path, 'pc1.npy'))
        pc2 = np.load(os.path.join(path, 'pc2.npy'))

        # why
        pc1[..., -1] *= -1
        pc2[...,-1] *= -1
        pc1[..., 0] *= -1
        pc2[..., 0] *= -1

        return pc1, pc2
