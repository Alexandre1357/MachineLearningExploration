import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import numpy as np

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class PBRDataset(Dataset):
    def __init__(self, albedo_img, arm_img, normal_img, sidelength):
        super().__init__()

        self.sidelength = sidelength
        self.size = sidelength * sidelength

        transform = Resize(self.sidelength)
        
        self.albedo_img = transform(albedo_img)
        self.arm_img = transform(arm_img)
        self.normal_img = transform(normal_img)

        albedo_data = np.asarray(transform(albedo_img), dtype=np.float32)
        arm_data = np.asarray(transform(arm_img), dtype=np.float32)
        normal_data = np.asarray(transform(normal_img), dtype=np.float32)

        combined = np.concatenate((albedo_data, arm_data, normal_data), axis=2)
        combined = np.reshape(combined, (sidelength * sidelength, 3 * 3))
        combined = combined / 255
        
        self.output_features_tensor = torch.from_numpy(combined).cuda()
        self.input_features_tensor = get_mgrid(sidelength).cuda()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx > self.size or idx < 0: raise IndexError

        return self.input_features_tensor[idx], self.output_features_tensor[idx]