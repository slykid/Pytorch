import os
from PIL import Image
import numpy as np
from glob import glob

from torch.utils.data import Dataset

class DogCatDataset(Dataset):
    def __init__(self, root_path, size=(224, 224)):
        self.root_path = root_path
        self.size = size

        self.files = glob(os.path.join(self.root_path, "*.jpg"))
        self.size = len(self.files)

        for t in ['train', 'valid']:
            for folder in ['dog', 'cat']:
                if os.path.isdir(os.path.join(self.root_path, t, folder)) is False:
                    os.makedirs(os.path.join(self.root_path, t, folder))

        if self.size > 0:
            shuffle = np.random.permutation(self.size)
            # valid
            for i in shuffle[:2000]:
                folder = self.files[i].split('\\')[-1].split('.')[0]
                image = self.files[i].split('\\')[-1]
                os.rename(self.files[i], os.path.join(self.root_path, 'valid', folder, image))

            # train
            for i in shuffle[2000:]:
                folder = self.files[i].split('\\')[-1].split('.')[0]
                image = self.files[i].split('\\')[-1]
                os.rename(self.files[i], os.path.join(self.root_path, 'train', folder, image))


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img = np.asarray(Image.open(self.files[index]).resize(self.size))
        label = self.files[index].split("\\")[-2]
        return img, label
