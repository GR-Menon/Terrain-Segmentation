from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path


class TerSegDataset(Dataset):
    """
    Multiclass Terrain Segmentation Dataset (https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022)
    """

    def __init__(self, img_paths, mask_paths, root=None, transforms=None):
        self.transforms = transforms
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.root = root

    def __len__(self):
        # return len(self.)
        pass

    # def convert_imgs(self, root):
    #     if not os.path.exists(root):
    #         os.mkdir(png_paths)
    #     for file in self.img_paths
    #
    # def convert_masks(self, root):

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transforms:
            img, mask = self.transforms(img), self.transforms(mask)
        mask = mask.squeeze(0)
        return img, mask
