import os
import numpy as np
from glob import glob
from PIL import Image

from torch.utils.data import Dataset

class cat_dog_dataset(Dataset):
    def __init__(self, root_dir, size=(224, 224)):
        self.files = glob(os.path.join(root_dir, "*/*.jpg"))
        self.size = size

        # 학습 데이터와 검증 데이터 분할
        if len(self.files) > 0:
            shuffle = np.random.permutation(len(self.files))

            os.mkdir(os.path.join(root_dir, 'valid'))

            for i in ['train', 'valid']:
                for folder in ['dog\\', 'cat\\']:
                    os.mkdir(os.path.join(root_dir, i, folder))

            # valid
            for i in shuffle[:2000]:
                folder = self.files[i].split('\\')[-1].split('.')[0]
                image = self.files[i].split('\\')[-1]
                os.rename(self.files[i], os.path.join(root_dir, 'valid', folder, image))

            # train
            for i in shuffle[2000:]:
                folder = self.files[i].split('\\')[-1].split('.')[0]
                image = self.files[i].split('\\')[-1]
                os.rename(self.files[i], os.path.join(root_dir, 'train', folder, image))
        else:
            print("files size is 0")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = np.asarray(Image.open(self.files[idx]).resize(self.size))
        label = self.files[idx].split('/')[-2]

        return image, label
