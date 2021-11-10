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
