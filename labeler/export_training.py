"""Write a labelled folder out as an image/mask pair set a trainer can read.

Two directories come out of it:

    <out>/images/<stem>.png   the source frame as 8-bit PNG
    <out>/labels/<stem>.png   an 8-bit PNG whose pixel values *are* class indices
    <out>/classes.txt         which index is which class

The label PNG is index-coded, not colour-coded: background is 0 and each class
takes the number of its row in the Classes panel, starting at 1. That panel's
order is the project's priority order, so where two classes overlap the one
further down wins — the same precedence the canvas and the saved JSON use.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGES_DIR = "images"
LABELS_DIR = "labels"
CLASSES_FILE = "classes.txt"

# An index-coded 8-bit mask cannot express more classes than this.
MAX_CLASSES = 255


def to_8bit(image: np.ndarray, auto_contrast: bool = False) -> np.ndarray:
    """An 8-bit copy of image, whatever depth it arrived in.

    The default is a plain full-range rescale, which is deterministic and maps
    every file the same way — two frames of the same scene keep their relative
    brightness, which matters for a training set. auto_contrast instead stretches
    each image's own 0.5-99.5 percentile range, which rescues data that occupies
    a fraction of its container (12-bit sensor output stored in 16-bit, say) at
    the cost of treating each image differently.
    """
    if image.dtype == np.uint8 and not auto_contrast:
        return image

    data = image.astype(np.float32)
    if auto_contrast:
        finite = data[np.isfinite(data)]
        if finite.size:
            lo, hi = np.percentile(finite, (0.5, 99.5))
        else:
            lo, hi = 0.0, 1.0
    elif image.dtype == np.uint16:
        lo, hi = 0.0, 65535.0
    elif image.dtype == np.int16:
        lo, hi = -32768.0, 32767.0
    elif image.dtype == np.uint8:
        lo, hi = 0.0, 255.0
    else:
        # Float or anything else carries no implied range, so its own extent is
        # the only thing available to map from.
        finite = data[np.isfinite(data)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    # Rounded, not truncated: truncation biases every value half a level down,
    # so a true mid-grey would come out 127 instead of 128.
    scaled = np.rint((data - lo) * (255.0 / (hi - lo)))
    return np.clip(np.nan_to_num(scaled), 0, 255).astype(np.uint8)


def normalise_channels(image: np.ndarray) -> np.ndarray:
    """Drop an alpha channel and collapse anything exotic to 1 or 3 channels."""
    if image.ndim == 2:
        return image
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return image
    if channels == 4:
        return image[:, :, :3]        # BGRA from a PNG/TIFF with alpha
    return image[:, :, :3] if channels > 4 else image[:, :, :1][:, :, 0]


def read_source(path: str) -> Optional[np.ndarray]:
    """Read an image at its true bit depth. None when the file is unreadable."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return normalise_channels(image)


def fit_size(src_w: int, src_h: int, width: Optional[int] = None,
             height: Optional[int] = None) -> Tuple[int, int]:
    """Target size with the source aspect ratio kept, driven by whichever side is given."""
    if src_w <= 0 or src_h <= 0:
        return max(1, width or 1), max(1, height or 1)
    if width and not height:
        return max(1, width), max(1, int(round(width * src_h / src_w)))
    if height and not width:
        return max(1, int(round(height * src_w / src_h))), max(1, height)
    return max(1, width or src_w), max(1, height or src_h)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a picture. Area for shrinking, linear for growing."""
    w, h = size
    if (image.shape[1], image.shape[0]) == (w, h):
        return image
    shrinking = w * h < image.shape[1] * image.shape[0]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (w, h), interpolation=interp)


def resize_label(label: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an index mask. Nearest only — averaging two class ids invents a third."""
    w, h = size
    if (label.shape[1], label.shape[0]) == (w, h):
        return label
    return cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)


def build_label(shape: Tuple[int, int], annotations: Sequence,
                class_index: Dict[int, int]) -> np.ndarray:
    """Index-coded mask for one image.

    annotations are painted in list order, which the app keeps sorted by the
    Classes panel, so a class further down the panel overwrites one above it
    where they overlap.
    """
    height, width = shape
    out = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        index = class_index.get(ann.cat_id)
        # A mask sized against a different image cannot be indexed into this
        # one; skipping beats raising and losing the rest of the export.
        if index is None or ann.mask.shape != out.shape:
            continue
        out[ann.mask > 0] = index
    return out


def class_index_map(categories: Sequence) -> Dict[int, int]:
    """cat_id → 1-based index, in Classes panel order. 0 is left for background."""
    return {cat.id: i for i, cat in enumerate(categories, start=1)}


def write_classes_file(path: str, categories: Sequence) -> None:
    """The legend for the label PNGs — without it the numbers mean nothing."""
    lines = ["0\tbackground"]
    lines += [f"{i}\t{cat.name}" for i, cat in enumerate(categories, start=1)]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def export(out_dir: str, image_dir: str, files: Sequence[str],
           categories: Sequence, annotations_for: Callable[[str], Sequence],
           width: Optional[int] = None, height: Optional[int] = None,
           auto_contrast: bool = False,
           progress: Optional[Callable[[int, int], None]] = None
           ) -> Tuple[int, List[str]]:
    """Write every file out as an image/label pair. Returns (written, problems).

    annotations_for(filename) supplies that image's annotations; the caller
    keeps them, and passing a callable means a file with none still exports with
    an all-background label rather than being skipped.

    width/height are the target size. Giving only one keeps the aspect ratio;
    giving both uses them as they are; giving neither keeps each source's size.
    """
    images_out = os.path.join(out_dir, IMAGES_DIR)
    labels_out = os.path.join(out_dir, LABELS_DIR)
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    index_of = class_index_map(categories)
    written = 0
    problems: List[str] = []
    total = max(1, len(files))

    for done, name in enumerate(files, start=1):
        try:
            source = read_source(os.path.join(image_dir, name))
            if source is None:
                problems.append(f"{name}: 읽을 수 없습니다")
                continue
            picture = to_8bit(source, auto_contrast)
            label = build_label(picture.shape[:2], annotations_for(name), index_of)

            size = fit_size(picture.shape[1], picture.shape[0], width, height)
            picture = resize_image(picture, size)
            # Resized from the same source size with the same target, so the two
            # land on identical grids; nearest keeps the ids intact.
            label = resize_label(label, size)

            stem = os.path.splitext(name)[0]
            if not cv2.imwrite(os.path.join(images_out, f"{stem}.png"), picture):
                problems.append(f"{name}: 이미지 저장 실패")
                continue
            if not cv2.imwrite(os.path.join(labels_out, f"{stem}.png"), label):
                problems.append(f"{name}: 라벨 저장 실패")
                continue
            written += 1
        except Exception as exc:                      # one bad file must not
            problems.append(f"{name}: {exc}")         # abandon the rest
        finally:
            if progress is not None:
                progress(done, total)

    write_classes_file(os.path.join(out_dir, CLASSES_FILE), categories)
    return written, problems
