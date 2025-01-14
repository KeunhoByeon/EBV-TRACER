import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset


def resize_and_pad_image(img, target_size=(640, 640), keep_ratio=False, padding=False, interpolation=None):
    # 1) Calculate ratio
    old_size = img.shape[:2]
    if keep_ratio:
        ratio = min(float(target_size[0]) / old_size[0], float(target_size[1]) / old_size[1])
        new_size = tuple([int(x * ratio) for x in old_size])
    else:
        new_size = target_size

    # 2) Resize image
    if interpolation is None:
        interpolation = cv2.INTER_AREA if new_size[0] < old_size[0] else cv2.INTER_CUBIC
    img = cv2.resize(img.copy(), (new_size[1], new_size[0]), interpolation=interpolation)

    # 3) Pad image
    if padding:
        delta_w = target_size[1] - new_size[1]
        delta_h = target_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        if (isinstance(padding, list) or isinstance(padding, tuple)) and len(padding) == 3:
            value = padding
        else:
            value = [0, 0, 0]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)

    return img


class EBVGCDataset(Dataset):
    def __init__(self, samples, input_size: int = None, is_train=False):
        self.input_size = input_size
        self.is_train = is_train

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Augmentation setting
        self.affine_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45)))
        ], random_order=True)
        self.color_seq = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
            iaa.Sometimes(0.2, iaa.Cutout(nb_iterations=(1, 5), size=0.2, cval=255, squared=False)),
            iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.0, 1.0))),
            iaa.Sometimes(0.5, iaa.MultiplyHue((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.MultiplySaturation((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.2)),
            iaa.Sometimes(0.5, iaa.LogContrast((0.6, 1.4))),
        ], random_order=True)

        self.samples = samples

    def __getitem__(self, index):
        img_path, gt = self.samples[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)

        if self.is_train:
            img = self.affine_seq.augment_image(img)
            img = self.color_seq.augment_image(img)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        return img_path, img, gt

    def __len__(self):
        return len(self.samples)
