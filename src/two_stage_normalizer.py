import os

import cv2
import numpy as np
import openslide
from joblib import Parallel, delayed
from tiatoolbox.tools.stainnorm import MacenkoNormalizer
from tqdm import tqdm

from .utils import get_files


class TwoStageStainNormalizer:
    def __init__(self,
                 target_svs_dir, input_svs_dir,
                 input_patch_dir, output_patch_dir, output_thumbnail_dir,
                 thumbnail_size=2048, n_jobs=8):
        self.target_svs_dir = target_svs_dir
        self.input_svs_dir = input_svs_dir
        self.input_patch_dir = input_patch_dir
        self.output_patch_dir = output_patch_dir
        self.output_thumbnail_dir = output_thumbnail_dir
        self.target_thumbnail_dir = os.path.join(self.output_thumbnail_dir, "target")
        self.input_thumbnail_dir = os.path.join(self.output_thumbnail_dir, "input")
        self.normalized_thumbnail_dir = os.path.join(self.output_thumbnail_dir, "normalized")

        self.target_svs_paths = get_files(self.target_svs_dir, ext=".svs")
        self.input_svs_paths = get_files(self.input_svs_dir, ext=".svs")

        self.thumbnail_size = thumbnail_size
        self.n_jobs = n_jobs

        self.slide_shapes = {}

    def _extract_thumbnail(self, svs_path, save_path):
        slide = openslide.OpenSlide(svs_path)
        w_pixels, h_pixels = slide.level_dimensions[0]
        filename = os.path.basename(svs_path)
        file_id = filename.replace(os.path.splitext(filename)[-1], "")

        ratio = min(self.thumbnail_size / w_pixels, self.thumbnail_size / h_pixels)
        thumbnail_shape = (int(w_pixels * ratio), int(h_pixels * ratio))

        thumbnail = slide.get_thumbnail(thumbnail_shape)
        thumbnail = np.array(thumbnail)
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)

        filename = os.path.basename(svs_path)
        save_path = save_path.replace(os.path.splitext(filename)[-1], ".png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, thumbnail)

        return file_id, save_path, (w_pixels, h_pixels)

    def _extract_thumbnails(self):
        results = Parallel(n_jobs=self.n_jobs)(delayed(self._extract_thumbnail)(
            file_path, file_path.replace(self.target_svs_dir, self.target_thumbnail_dir)
        ) for file_path in tqdm(self.target_svs_paths))

        self.target_thumbnail_paths = []
        for file_id, save_path, (w_pixels, h_pixels) in results:
            self.target_thumbnail_paths.append(save_path)
            self.slide_shapes[file_id] = (w_pixels, h_pixels)

        results = Parallel(n_jobs=self.n_jobs)(delayed(self._extract_thumbnail)(
            file_path, file_path.replace(self.input_svs_dir, self.input_thumbnail_dir)
        ) for file_path in tqdm(self.input_svs_paths))

        self.input_thumbnail_paths = []
        for file_id, save_path, (w_pixels, h_pixels) in results:
            self.input_thumbnail_paths.append(save_path)
            self.slide_shapes[file_id] = (w_pixels, h_pixels)

    def _normalize_thumbnail(self, input_path, save_path, normalizer):
        source = cv2.imread(input_path, cv2.IMREAD_COLOR)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        transformed = normalizer.transform(source)
        transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, transformed)

        filename = os.path.basename(input_path)
        file_id = filename.replace(os.path.splitext(filename)[-1], "")
        return file_id, save_path

    def _normalize_thumbnails(self):
        all_target_thumbnails = []
        taget_thumbnail_size = self.thumbnail_size // len(self.target_svs_paths)
        for target_thumbnail_path in self.target_thumbnail_paths:
            target_thumbnail = cv2.imread(target_thumbnail_path)
            h, w = target_thumbnail.shape[:2]
            ratio = taget_thumbnail_size / w
            target_thumbnail = cv2.resize(target_thumbnail, (int(h * ratio), taget_thumbnail_size))
            target_thumbnail = cv2.cvtColor(target_thumbnail, cv2.COLOR_BGR2RGB)
            all_target_thumbnails.append(target_thumbnail)
        all_target_thumbnails = np.concatenate(all_target_thumbnails, axis=1)

        normalizer = MacenkoNormalizer()
        normalizer.fit(all_target_thumbnails)

        normalized_thumbnail_paths = Parallel(n_jobs=self.n_jobs)(delayed(self._normalize_thumbnail)(
            input_path, input_path.replace(self.input_thumbnail_dir, self.normalized_thumbnail_dir), normalizer
        ) for input_path in tqdm(self.input_thumbnail_paths))

        self.normalized_thumbnail_paths = {}
        for file_id, file_path in normalized_thumbnail_paths:
            self.normalized_thumbnail_paths[file_id] = file_path

    def _get_patch_paths(self):
        self.patch_paths = {}

        all_patch_paths = get_files(self.input_patch_dir, ext=".png")
        for patch_path in all_patch_paths:
            file_id = os.path.basename(os.path.dirname(patch_path))
            if file_id not in self.patch_paths:
                self.patch_paths[file_id] = []
            self.patch_paths[file_id].append(patch_path)

    def _normalize_patch_in_silde(self, patch_path, thumbnail_patch):
        patch = cv2.imread(patch_path)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        normalizer = MacenkoNormalizer()
        normalizer.fit(thumbnail_patch)
        patch_transformed = normalizer.transform(patch)

        save_path = patch_path.replace(self.input_patch_dir, self.output_patch_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        patch_transformed = cv2.cvtColor(patch_transformed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, patch_transformed)

    def _normalize_patch_in_sildes(self):
        for i, file_id in enumerate(self.patch_paths.keys()):
            normalized_thumbnail_path = self.normalized_thumbnail_paths[file_id]
            normalized_thumbnail = cv2.imread(normalized_thumbnail_path)
            normalized_thumbnail = cv2.cvtColor(normalized_thumbnail, cv2.COLOR_BGR2RGB)
            w_pixels, h_pixels = self.slide_shapes[file_id]

            patch_queue = []
            for patch_path in self.patch_paths[file_id]:
                coords = patch_path.split('_patch_')[-1].split('.')[0].split('_')
                y1, x1, y2, x2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                y1 = int(y1 * normalized_thumbnail.shape[0] / h_pixels)
                y2 = int(y2 * normalized_thumbnail.shape[0] / h_pixels)
                x1 = int(x1 * normalized_thumbnail.shape[1] / w_pixels)
                x2 = int(x2 * normalized_thumbnail.shape[1] / w_pixels)
                patch_queue.append([patch_path, y1, x1, y2, x2])

            Parallel(n_jobs=self.n_jobs)(delayed(self._normalize_patch_in_silde)(
                patch_path, normalized_thumbnail[y1:y2, x1:x2]
            ) for patch_path, y1, x1, y2, x2 in tqdm(patch_queue, desc="[{}/{}] {}".format(
                i, len(self.patch_paths), os.path.basename(file_id)
            )))

    def two_stage_normalize(self):
        print("Extracting thumbnails...")
        self._extract_thumbnails()
        print("Normalizing thumbnails...")
        self._normalize_thumbnails()
        print("Loading patch paths...")
        self._get_patch_paths()
        print("Normalizing patches...")
        self._normalize_patch_in_sildes()
