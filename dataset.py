import random

import torch
from PIL import Image


def balanced_split(root_path, val_ratio, class_map):
    # Group all filenames by class
    class_dict = {k:[] for k in class_map}
    for p in root_path.glob("*/*"):
        class_name = p.parent.stem
        class_dict[class_name].append(p)

    # Randomly split each class individually to ensure balance in both splits
    train_list = []
    val_list = []
    for class_name, img_paths in class_dict.items():
        random.Random(0).shuffle(img_paths)
        split_idx = val_ratio * len(img_paths)
        split_idx = int(val_ratio * len(img_paths))
        val_list += img_paths[:split_idx]
        train_list += img_paths[split_idx:]

    # Shuffle final lists to separate classes
    random.Random(0).shuffle(train_list)
    random.Random(0).shuffle(val_list)

    return train_list, val_list


class MoneyDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, root, split_list, img_transforms, class_map):
        self.root = root
        self.dataset = split_list
        self.img_transforms = img_transforms
        self.class_map = class_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename = self.dataset[idx]
        class_name = filename.parent.stem
        class_id = self.class_map.index(class_name)

        img = Image.open(filename)
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        return (img, class_id)


if __name__=="__main__":
    from pathlib import Path
    from torchvision import transforms as tf
    from matplotlib import pyplot as plt
    from constants import CLASS_MAP
    import utils

    input_size = (256, 256)
    train_transforms = tf.Compose([
        tf.RandomRotation(degrees=180, interpolation=tf.InterpolationMode.BILINEAR),
        tf.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        tf.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5)),
        tf.ColorJitter(brightness=0.7, contrast=0.6, saturation=0.7, hue=0.1),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = tf.Compose([
        tf.Resize(input_size),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Get file lists for train/val split
    val_ratio = 0.25
    root_path = Path('/home/martin/datasets/Money_Classification')
    train_list, val_list = balanced_split(root_path, val_ratio, CLASS_MAP)
    
    # Create datasets and data loaders
    train_set = MoneyDataset(root_path, train_list, train_transforms, CLASS_MAP)
    val_set = MoneyDataset(root_path, val_list, val_transforms, CLASS_MAP)

    print(f"Length of training set: {len(train_set)}")
    print(f"Length of validation set: {len(val_set)}")
    print(f"Validation ratio: {len(val_set)/(len(train_set)+len(val_set))}")

    sample = train_set[0]
    print(f"Image shape: {sample[0].shape}")
    print(f"Image dtype: {sample[0].dtype}")
    print(f"Class label: {CLASS_MAP[sample[1]]}")
    plt.imshow(utils.tensor_to_np(sample[0]))
    plt.show()
