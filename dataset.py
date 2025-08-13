import torch
import numpy as np
from torch.utils.data import Dataset

class UltrasoundNpyDataset_NoTransforms(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image_np = self.x_data[idx]
        mask_np = self.y_data[idx]

        image_tensor = torch.from_numpy(image_np).float()
        if image_tensor.ndim == 3 and image_tensor.shape[-1] in [1, 3]:
            image_tensor = image_tensor.permute(2, 0, 1)
        elif image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)

        if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            mask_np = np.squeeze(mask_np, axis=-1)
        mask_tensor = torch.from_numpy(mask_np).long()
        mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor
