import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from terrain_segmentation.formfactor import IMG_PATHS, MASK_PATHS, ROOT
from terrain_segmentation.utils import get_paths


class TerSegDataset(Dataset):
    """
    Multiclass Terrain Segmentation Dataset (https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022)
    """

    def __init__(self, root, image_dir, mask_dir, transforms=None, png: bool = False):
        self.transforms = transforms
        self.root = root
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = None
        self.mask_paths = None

        if png:
            print("Converting files to png format ...")
            self.convert_to_png()
        else:
            self.image_paths = get_paths(self.image_dir)
            self.mask_paths = get_paths(self.mask_dir)

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def convert_to_png(self):
        png_img_path = self.root / "png_gt"
        png_mask_path = self.root / "png_mask"
        if not os.path.exists(png_img_path) and not os.path.exists(png_mask_path):
            os.mkdir(png_img_path)
            os.mkdir(png_mask_path)
            print("Created png image and mask directories ...")

        for file in os.listdir(self.image_dir):
            img = Image.open(self.image_dir / file)
            img.save(f"{png_img_path}/{file[:-4]}.png")
        for file in os.listdir(self.mask_dir):
            mask = Image.open(self.mask_dir / file)
            mask.save(f"{png_mask_path}/{file[:-4]}.png")
        self.image_paths = get_paths(png_img_path)
        self.mask_paths = get_paths(png_mask_path)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transforms:
            img, mask = self.transforms(img), self.transforms(mask)
        mask = mask.squeeze(0)
        return img, mask


def test_dataset():
    transforms = [T.ToTensor()]
    dset = TerSegDataset(
        root=ROOT,
        image_dir=IMG_PATHS,
        mask_dir=MASK_PATHS,
        transforms=transforms,
        png=True,
    )
    image, mask = dset[0]
    print(image.shape, mask.shape)


if "__main__" == __name__:
    test_dataset()
