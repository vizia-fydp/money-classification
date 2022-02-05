import torch
from PIL import Image


class MoneyDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, root, split_idx, img_transforms, class_map):
        self.root = root
        self.split_idx = split_idx
        self.img_transforms = img_transforms
        self.class_map = class_map
        self.dataset = list(root.glob('*/*'))

    def __len__(self):
        return self.split_idx.shape[0]

    def __getitem__(self, idx):
        filename = self.dataset[self.split_idx[idx]]
        class_name = int(filename.parent.stem)
        class_id = self.class_map.index(class_name)

        img = Image.open(filename)
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        return (img, class_id)


if __name__=="__main__":
    from pathlib import Path
    from torchvision import transforms as tf
    import numpy as np
    from constants import CLASS_MAP

    input_size = 384
    img_transforms = tf.Compose([
        tf.RandomRotation(degrees=180, interpolation=tf.InterpolationMode.BILINEAR),
        tf.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        tf.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5)),
        tf.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.6, hue=0.05),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Don't split into train/val, just for testing
    root_path = Path('/home/martin/datasets/Money_Classification')
    ds_size = len(list(root_path.glob('*/*')))
    idx = np.arange(ds_size)

    ds = MoneyDataset(root_path, idx, img_transforms, CLASS_MAP)
    print(len(ds))
    sample = ds[0]
    print(sample[0].shape)
    print(sample[0].dtype)
    print(CLASS_MAP[sample[1]])
