import os
from PIL import Image
import numpy as np
from glob import glob

from torch.utils.data import Dataset

class DogCatDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform

        for t in ['train', 'valid']:
            for folder in ['dog', 'cat']:
                # 폴더가 없는 경우
                if os.path.isdir(os.path.join(self.root_path, t, folder)) is False:
                    os.makedirs(os.path.join(self.root_path, t, folder))
                    self.files = glob(os.path.join(self.root_path, "*.jpg"))
                    self.size = len(self.files)
                    print(self.size)

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

                # 폴더가 생성된 경우
                else:
                    self.files = glob(os.path.join(self.root_path, t, "*", "*.jpg"))
                    self.size = len(self.files)
                    print(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        label = self.files[index].split('\\')[-1].split('.')[0]

        if self.transform:
            img = self.transform(img)

        return img, label
