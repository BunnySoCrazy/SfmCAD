import numpy as np
import os
import torch
import torch.utils.data
import h5py

class BSP_GTSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        grid_sample=64,
        test_flag=False,
        shapenet_flag = False,
        cat = ''
    ):
        print('data source', data_source)
        self.data_source = data_source
        cat = data_source.split('/')[-1].split('_')[0]
        if test_flag:
            filename_shapes = os.path.join(self.data_source, cat + '_vox256_img_test.hdf5')
            name_file = os.path.join(self.data_source, cat + '_vox256_img_test.txt')
            npz_shapes = np.genfromtxt(name_file, dtype='str', delimiter='\n')
            self.data_names = npz_shapes
        else:
            filename_shapes = os.path.join(self.data_source, cat + '_vox256_img_train.hdf5')
            name_file = os.path.join(self.data_source, cat + '_vox256_img_train.txt')
            npz_shapes = np.genfromtxt(name_file, dtype='str', delimiter='\n')
            self.data_names = npz_shapes

        data_dict = h5py.File(filename_shapes, 'r')
        print(data_dict.keys())
        print('grid_sample',grid_sample)

        data_voxels = torch.from_numpy(data_dict['voxels'][:]).float()
        self.data_voxels = data_voxels.squeeze(-1).unsqueeze(1)
        data_points = torch.from_numpy(data_dict['points_'+str(grid_sample)][:]).float()
        data_values = torch.from_numpy(data_dict['values_'+str(grid_sample)][:]).float()

        if data_values.dim() < data_points.dim():
            data_values = data_values.unsqueeze(-1)
            data_points[torch.isnan(data_points)] = 0
            data_values[torch.isnan(data_values)] = 0
            self.data_points = torch.cat([data_points, data_values], 2)
            perm = torch.randperm(self.data_points.size(1))
            self.data_points = self.data_points[:, perm, :]
        else:
            data_points = (data_points+0.5)/256-0.5
            self.data_points = torch.cat([data_points, data_values], 2)

        print('Loaded voxels shape, ', self.data_voxels.shape)
        print('Loaded points shape, ', self.data_points.shape)

    def __len__(self):
        return len(self.data_voxels)

    def __getitem__(self, idx):
        return {"voxels":self.data_voxels[idx], "occ_data":self.data_points[idx]}
