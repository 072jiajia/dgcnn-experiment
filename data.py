import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
def load_mask(partition, permute):
    raw_path='data/h5_files/main_split'
    data = h5py.File("{}/{}_objectdataset_{}.h5".format(raw_path, partition, permute), 'r+')
    mask = data['mask']
    mask[mask != -1] = 1
    mask[mask == -1] = 0 
    return mask
    
class ScanObject_coseg(Dataset):
    """Scan object dataset
    permute: ['augmented25rot', 'augmented25_norot', 'augmentedrot', 'augmentedrot_scale75', 'raw']
    obj: {0: 'bag', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk',
          6: 'display', 7: 'door', 8: 'shelf', 9: 'table', 10: 'bed', 11: 'pillow',
          12: 'sink', 13: 'sofa', 14: 'toilet', 15: 'all'}
    """
    def __init__(self,raw_path='data/h5_files/main_split', n_points=1024, 
             partition='training', permute='raw', obj=15, label_binarize=True, norm=True, center=True):       
        cat_to_label = {0: 'bag', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk',
                        6: 'display', 7: 'door', 8: 'shelf', 9: 'table', 10: 'bed', 11: 'pillow',
                        12: 'sink', 13: 'sofa', 14: 'toilet', 15: 'all'}

        self.partition = partition
        self.n_points = n_points
        self.norm = norm
        self.center = center

        if permute == 'raw':
            data = h5py.File("{}/{}_objectdataset.h5".format(raw_path, partition), 'r+')
        else:
            data = h5py.File("{}/{}_objectdataset_{}.h5".format(raw_path, partition, permute), 'r+')    
            
        self.points = np.array(data['data'][:].astype('float32'))
        self.labels = np.array(data['label'][:].astype('int64'))
        self.masks = np.array(data['mask'][:].astype('float32'))
        
        data.close()
        
        select_label = cat_to_label[obj]

        if select_label != 'all':
            self.points = self.points[self.labels == obj]
            self.masks = self.masks[self.labels == obj]
            self.labels = self.labels[self.labels == obj]
        if label_binarize:
            self.masks[self.masks != -1] = 1
            self.masks[self.masks == -1] = 0  
        else:
            self.masks[:, :] = 1 # 0 stands for BG
        
        # self.points = normalize(self.points)
            
        print("Number of data:", self.points.shape[0])
        
    def __getitem__(self, item):
        coord = self.points[item][:self.n_points]
        label = self.labels[item]
        mask = self.masks[item][:self.n_points]

        if self.partition == 'training':
            # Shuffle the points to do random sampling
            perm_index = np.random.permutation(self.points.shape[1])
            self.points[item] = self.points[item, perm_index]
            self.masks[item] = self.masks[item, perm_index]

        if self.partition == 'training':
            coord = translate_pointcloud(coord)
        if self.partition == 'test':
            return self.points[item],self.labels[item],self.masks[item]
        else:
            return coord, label, mask
   

    def __len__(self):
        return len(self.points)


def normalize(PointClouds, epsilon=1e-8):
    # Not Suggest
    Centers = np.mean(PointClouds, axis=1, keepdims=True)
    PointClouds -= Centers
    Vars = np.mean(PointClouds ** 2, axis=(1, 2), keepdims=True) + epsilon
    PointClouds /= np.sqrt(Vars)
    return PointClouds


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    #print(translated_pointcloud.shape)
    return translated_pointcloud

class TrainingBatchSampler(BatchSampler):
    ''' Batch Sampler
    '''

    def __init__(self, dataset, n_points_choices, batch_size):
        self.dataset = dataset
        self.choices = n_points_choices
        self.batch_size = batch_size

    def __iter__(self):
        ''' Randomly set the number of points '''
        # initialize yield count and object indices
        count = 0
        obj_idx = np.random.permutation(len(self.dataset))
        while count + self.batch_size < len(self.dataset):
            # Randomly set the number of sample points
            n_points = np.random.choice(self.choices, 1)[0]
            self.dataset.n_points = n_points
            
            # yield the index of object to data_loader
            yield obj_idx[count: count + self.batch_size]
            count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size