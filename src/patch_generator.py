import cv2
import numpy as np
import openslide
import os
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import get_files


class PatchGenerator:
    def __init__(self, patch_size, save_size, patch_step=1.0,
                 min_red=64, max_green=230, threshold=0.5, n_jobs=8):
        self.patch_size = patch_size
        self.save_size = save_size
        self.patch_step = patch_step

        self.min_red = min_red
        self.max_green = max_green
        self.threshold = threshold

        self.n_jobs = n_jobs

    def _is_background(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        slide_white = np.where(img[:, :, 1] > self.max_green, 1, 0)
        slide_black = np.where(img[:, :, 2] < self.min_red, 1, 0)
        if np.mean(np.where(slide_white + slide_black > 0, 1, 0)) > self.threshold:
            return True
        return False

    def _generate_patch(self, patch, svs_path, h_i, w_i, save_dir):
        patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
        patch = cv2.resize(patch, (self.save_size, self.save_size))
        if self._is_background(patch):
            return

        filename = os.path.basename(svs_path)
        file_id = filename.replace(os.path.splitext(filename)[-1], "")
        slide_save_filename = '{}_patch_{}_{}_{}_{}.png'.format(
            file_id, h_i, w_i, h_i + self.patch_size, w_i + self.patch_size)
        save_path = os.path.join(save_dir, slide_save_filename)
        cv2.imwrite(save_path, patch)

    def generate_patches(self, svs_dir, base_save_dir):
        svs_paths = get_files(svs_dir, ".svs")

        for i, svs_path in tqdm(enumerate(svs_paths)):
            slide = openslide.OpenSlide(svs_path)
            w_pixels, h_pixels = slide.level_dimensions[0]

            coords = []
            for w_i in range(0, w_pixels - self.patch_size, int(self.patch_size * self.patch_step)):
                for h_i in range(0, h_pixels - self.patch_size, int(self.patch_size * self.patch_step)):
                    coords.append((w_i, h_i))

            filename = os.path.basename(svs_path)
            save_dir = svs_path.replace(svs_dir, base_save_dir)
            save_dir = save_dir.replace(os.path.splitext(filename)[-1], "")
            os.makedirs(save_dir, exist_ok=True)
            print(save_dir)

            Parallel(n_jobs=self.n_jobs)(delayed(self._generate_patch)(
                slide.read_region((w_i, h_i), 0, (self.patch_size, self.patch_size)),
                svs_path, h_i, w_i, save_dir
            ) for w_i, h_i in tqdm(coords, desc="[{}/{}] {}".format(i, len(svs_paths), os.path.basename(svs_path))))
