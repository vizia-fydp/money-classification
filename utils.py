import numpy as np


def tensor_to_np(tensor):
    """
    Denormalize tensor and convert from (C, H, W) float to (H, W, C) uint8 np array for displaying.
    """
    # From (C, H, W) to (H, W, C)
    np_img = tensor.permute(1, 2, 0).numpy()

    # Rescale to 0-1
    np_img -= np_img.min()
    np_img *= 255 / np_img.max()
    np_img = np_img.astype(np.uint8)

    return np_img
