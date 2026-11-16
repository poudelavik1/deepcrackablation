"""Image/mask tile pairs, the fixed augmentation, and loaders.

Every model receives RGB float tensors in [0, 1]; normalization is part of the model
(see models/registry.py) because the ported networks expect raw [0, 1] input.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

cv2.setNumThreads(0)  # the DataLoader workers already parallelize

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}
TILE_PATTERN = re.compile(r"^(?P<source>.*?)[_\-]r(?:ow)?[_\-]?\d+[_\-]c(?:ol)?[_\-]?\d+$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def normalized_stem(path: Path) -> str:
    """Lower-case stem without role suffixes (_mask, _label, _image, _img, _gt)."""
    stem = path.stem.lower().strip()
    previous = None
    while stem != previous:
        previous = stem
        stem = re.sub(r"(?:[_\- ](?:mask|label|image|img|gt))$", "", stem)
    return stem


def source_group(stem: str) -> str:
    """Source of a tile: the prefix of <source>_r####_c####, else the stem with digits as '#'."""
    match = TILE_PATTERN.match(stem)
    if match:
        return match.group("source") or "tiles"
    return re.sub(r"\d+", "#", re.sub(r"(?:[_\-]\d+)+$", "", stem)) or "#"


def split_dir(root: Path, split: str) -> Path:
    for name in SPLIT_ALIASES[split]:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == name:
                return child
    raise FileNotFoundError(f"No {split} folder under {root}")


def collect_pairs(root: Path, split: str, exclude_regex: list[str] | None = None) -> list[tuple[str, Path, Path]]:
    """Sorted (name, image path, mask path) of one split, paired by normalized stem."""
    folder = split_dir(root, split)
    image_dir, mask_dir = folder / "images", folder / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing images/ or masks/ in {folder}")

    def index(directory: Path) -> dict[str, Path]:
        return {normalized_stem(p): p for p in sorted(directory.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}

    images, masks = index(image_dir), index(mask_dir)
    missing = sorted(set(images) ^ set(masks))
    if missing:
        raise ValueError(f"{len(missing)} unpaired files in {folder}, e.g. {missing[:5]}")
    patterns = [re.compile(p, re.IGNORECASE) for p in (exclude_regex or [])]
    return [(key, images[key], masks[key]) for key in sorted(images)
            if not any(p.search(key) for p in patterns)]


def read_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 image and {0,1} uint8 mask of identical size."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or mask is None:
        raise ValueError(f"Could not read pair: {image_path}, {mask_path}")
    if image.shape[:2] != mask.shape:
        # Phone photos may carry an EXIF rotation that the mask did not get.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image.shape[:2] != mask.shape:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    # JPEG masks (CRACK500) carry compression noise; 0/1 masks also occur.
    threshold = 1 if mask.max() <= 1 else 128
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (mask >= threshold).astype(np.uint8)


def mask_size(mask_path: Path) -> tuple[int, int]:
    """(height, width) from the file header."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(mask_path) as image:
        return image.size[1], image.size[0]


def subsample(items: list, limit: int) -> list:
    """Evenly spaced subset of `limit` items (deterministic); all items if limit is 0."""
    if not limit or limit >= len(items):
        return items
    positions = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in sorted(set(positions.tolist()))]


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def _transform(cls, p: float, *alternatives: dict[str, Any]):
    """`cls(p=p, **options)` with the first option set this albumentations version accepts.

    Several transforms renamed their arguments between 1.4 and 2.x (the server and the
    laptop may differ), so each new-style set is followed by its old-style equivalent.
    """
    import inspect

    known = set(inspect.signature(cls.__init__).parameters)
    for options in alternatives or ({},):
        if set(options) <= known:
            return cls(p=p, **options)
    raise TypeError(f"{cls.__name__}: no argument set fits this albumentations version ({sorted(known)})")


def build_train_transform(config: dict[str, Any], seed: int | None = None):
    """Geometric + mild photometric augmentation shared by all models.

    The benchmark uses flips, 90-degree rotations, a mild affine, brightness/contrast, HSV and
    light blur/noise. The steps after those are off (p = 0) in the benchmark and exist for the
    augmentation ablation; CutMix works on the batch and is applied in train.py.
    """
    import albumentations as A

    aug, crop = config["aug"], int(config["data"]["crop_size"])
    steps: list[Any] = [A.PadIfNeeded(min_height=crop, min_width=crop, border_mode=cv2.BORDER_REFLECT_101)]
    if aug.get("crop", "crack") == "random":
        steps.append(A.RandomCrop(height=crop, width=crop))
    else:
        # Large photos are mostly background: keep a crack in the crop when there is one.
        steps.append(A.CropNonEmptyMaskIfExists(height=crop, width=crop))
    if aug.get("flip", True):
        steps += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    if aug.get("rotate90", True):
        steps.append(A.RandomRotate90(p=0.5))
    if aug.get("affine_p", 0) > 0:
        steps.append(A.Affine(scale=tuple(aug["affine_scale"]), rotate=tuple(aug["affine_rotate"]), p=aug["affine_p"]))
    if aug.get("brightness_contrast_p", 0) > 0:
        steps.append(A.RandomBrightnessContrast(
            brightness_limit=aug["brightness_limit"], contrast_limit=aug["contrast_limit"],
            p=aug["brightness_contrast_p"]))
    if aug.get("hsv_p", 0) > 0:
        steps.append(A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=aug["hsv_p"]))
    if aug.get("blur_noise_p", 0) > 0:
        steps.append(A.OneOf([A.GaussianBlur(blur_limit=(3, 3)), A.GaussNoise()], p=aug["blur_noise_p"]))

    # ---- ablation-only steps (geometric, then photometric, then occlusion) ---- #
    def p(key: str) -> float:
        return float(aug.get(key, 0) or 0)

    if p("elastic_p") > 0:       # local warping of the crack path (masks use nearest-neighbour)
        steps.append(_transform(A.ElasticTransform, p("elastic_p"), {"alpha": 30.0, "sigma": 5.0}))
    if p("grid_distortion_p") > 0:
        steps.append(_transform(A.GridDistortion, p("grid_distortion_p"), {"num_steps": 5, "distort_limit": 0.2}))
    if p("perspective_p") > 0:   # oblique views
        steps.append(_transform(A.Perspective, p("perspective_p"), {"scale": (0.02, 0.06)}))
    if p("clahe_p") > 0:         # local contrast
        steps.append(_transform(A.CLAHE, p("clahe_p"), {"clip_limit": (1.0, 3.0), "tile_grid_size": (8, 8)}))
    if p("gamma_p") > 0:
        steps.append(_transform(A.RandomGamma, p("gamma_p"), {"gamma_limit": (80, 120)}))
    if p("shadow_p") > 0:        # cast shadows anywhere in the crop (the default only uses the lower half)
        steps.append(_transform(A.RandomShadow, p("shadow_p"), {"shadow_roi": (0, 0, 1, 1)}))
    if p("gray_p") > 0:
        steps.append(_transform(A.ToGray, p("gray_p")))
    if p("sharpen_p") > 0:
        steps.append(_transform(A.Sharpen, p("sharpen_p"), {"alpha": (0.1, 0.3), "lightness": (0.8, 1.2)}))
    if p("motion_blur_p") > 0:   # drone / camera motion
        steps.append(_transform(A.MotionBlur, p("motion_blur_p"), {"blur_limit": (3, 7)}))
    if p("downscale_p") > 0:     # lower ground sampling distance: down- and up-sample
        steps.append(_transform(A.Downscale, p("downscale_p"), {"scale_range": (0.5, 0.9)},
                                {"scale_min": 0.5, "scale_max": 0.9}))
    if p("iso_noise_p") > 0:     # camera sensor noise
        steps.append(_transform(A.ISONoise, p("iso_noise_p"), {"color_shift": (0.01, 0.05), "intensity": (0.1, 0.4)}))
    if p("jpeg_p") > 0:
        steps.append(_transform(A.ImageCompression, p("jpeg_p"), {"quality_range": (50, 95)},
                                {"quality_lower": 50, "quality_upper": 95}))
    if p("dropout_p") > 0:       # occlusion; the mask is cleared too (a hidden crack is not a crack to find)
        holes, size = (1, 4), (16, 48)
        steps.append(_transform(
            A.CoarseDropout, p("dropout_p"),
            {"num_holes_range": holes, "hole_height_range": size, "hole_width_range": size, "fill": 0, "fill_mask": 0},
            {"num_holes_range": holes, "hole_height_range": size, "hole_width_range": size, "fill_value": 0,
             "mask_fill_value": 0},
            {"min_holes": holes[0], "max_holes": holes[1], "min_height": size[0], "max_height": size[1],
             "min_width": size[0], "max_width": size[1], "fill_value": 0, "mask_fill_value": 0}))
    try:
        return A.Compose(steps, seed=seed)
    except TypeError:  # albumentations < 1.4.22 has no seed argument; it uses random / numpy.random
        return A.Compose(steps)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def to_tensors(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()
    return image_tensor, mask_tensor


class CrackDataset(Dataset):
    """Tile pairs of one split. With a transform: augmented crops; without: full tiles."""

    def __init__(self, root: Path, split: str, config: dict[str, Any], transform=None, limit: int = 0):
        self.pairs = subsample(collect_pairs(Path(root), split, config["data"].get("exclude_regex")), limit)
        if not self.pairs:
            raise ValueError(f"No tiles in the {split} split of {root}")
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def names(self) -> list[str]:
        return [name for name, _, _ in self.pairs]

    def sizes(self) -> list[tuple[int, int]]:
        return [mask_size(mask_path) for _, _, mask_path in self.pairs]

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, image_path, mask_path = self.pairs[index]
        image, mask = read_pair(image_path, mask_path)
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        image_tensor, mask_tensor = to_tensors(image, mask)
        return {"image": image_tensor, "mask": mask_tensor, "name": name}


class SizeBucketSampler(Sampler[list[int]]):
    """Evaluation batches of tiles that share one size, in a fixed order."""

    def __init__(self, sizes: list[tuple[int, int]], batch_size: int, max_side: int):
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, size in enumerate(sizes):
            buckets[size].append(index)
        self.batches: list[list[int]] = []
        for size, indices in sorted(buckets.items()):
            step = 1 if max(size) > max_side else batch_size  # sliding-window images go one at a time
            self.batches += [indices[i:i + step] for i in range(0, len(indices), step)]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "name": [item["name"] for item in batch],
    }


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def train_loader(root: Path, config: dict[str, Any], seed: int, workers: int, batch_size: int) -> DataLoader:
    dataset = CrackDataset(root, "train", config, build_train_transform(config, seed),
                           limit=int(config["data"].get("max_train_samples", 0)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) > batch_size,
        num_workers=workers, pin_memory=torch.cuda.is_available(), collate_fn=collate,
        worker_init_fn=seed_worker, generator=generator, persistent_workers=workers > 0,
    )


def eval_loader(root: Path, split: str, config: dict[str, Any], workers: int, batch_size: int | None = None) -> DataLoader:
    dataset = CrackDataset(root, split, config, None, limit=int(config["data"].get("max_eval_samples", 0)))
    sampler = SizeBucketSampler(
        dataset.sizes(), batch_size or int(config["train"]["eval_batch_size"]), int(config["data"]["eval_max_side"]))
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=collate)
