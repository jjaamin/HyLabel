from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .models import Project, ImageAnnotation, PALETTE
from .mask_manager import MaskManager

LABELME_VERSION = "1.0.1"
_DESCRIPTION = "label editor - jamin"


# ── Save ──────────────────────────────────────────────────────────────────────

def save_labelme(
    project: Project,
    mask_managers: Dict[int, MaskManager],
    image_dir: str,
) -> None:
    """Write one <stem>.json per image into image_dir (LabelMe format)."""
    cat_map = {c.id: c for c in project.categories}

    for img in project.images:
        shapes: list = []
        mgr = mask_managers.get(img.image_id)

        if mgr:
            for ann in mgr.annotations():
                cat = cat_map.get(ann.cat_id)
                if cat is None:
                    continue

                if ann.original_polygons is not None:
                    # Annotation unchanged since load — write original points as-is.
                    for poly_pts in ann.original_polygons:
                        if len(poly_pts) >= 3:
                            shapes.append({
                                "label": cat.name,
                                "points": poly_pts,
                                "group_id": None,
                                "description": _DESCRIPTION,
                                "shape_type": "polygon",
                                "flags": {},
                                "mask": None,
                            })
                else:
                    # Mask was brush/magic-edited — extract full-density contour.
                    contours, _ = cv2.findContours(
                        ann.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    for c in contours:
                        if len(c) < 3:
                            continue
                        pts = c.reshape(-1, 2).tolist()
                        shapes.append({
                            "label": cat.name,
                            "points": [[float(x), float(y)] for x, y in pts],
                            "group_id": None,
                            "description": _DESCRIPTION,
                            "shape_type": "polygon",
                            "flags": {},
                            "mask": None,
                        })

        img_basename = os.path.basename(img.file_path)
        stem = os.path.splitext(img_basename)[0]
        json_path = os.path.join(image_dir, stem + ".json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": LABELME_VERSION,
                "flags": {},
                "shapes": shapes,
                "imagePath": img_basename,
                "imageData": None,
                "imageHeight": img.height,
                "imageWidth": img.width,
            }, f, ensure_ascii=False, indent=2)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_labelme(
    json_dir: str,
    image_filenames: List[str],
) -> Tuple[Project, Dict[int, MaskManager]]:
    """
    Load LabelMe JSON files from json_dir that correspond to image_filenames.
    Returns (project, mask_managers).
    """
    project = Project()
    mask_managers: Dict[int, MaskManager] = {}
    label_to_cat: dict = {}

    # First pass — collect unique labels to build a stable category list
    raw: Dict[str, dict] = {}
    for fname in image_filenames:
        stem = os.path.splitext(fname)[0]
        json_path = os.path.join(json_dir, stem + ".json")
        if not os.path.isfile(json_path):
            continue
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        raw[fname] = data
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if label and label not in label_to_cat:
                cat = project.add_category(label)
                label_to_cat[label] = cat

    # Second pass — rasterize shapes per image
    for fname, data in raw.items():
        w = data.get("imageWidth", 0)
        h = data.get("imageHeight", 0)
        if w <= 0 or h <= 0:
            continue

        img_path = os.path.join(json_dir, fname)
        img_ann = project.get_or_create_image(img_path, w, h)
        mgr = MaskManager(w, h)

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                continue
            label = shape.get("label", "").strip()
            cat = label_to_cat.get(label)
            if cat is None:
                continue
            points = shape.get("points", [])
            if len(points) < 3:
                continue
            pts_float = [[float(x), float(y)] for x, y in points]
            mask = np.zeros((h, w), dtype=np.uint8)
            MaskManager.fill_polygon_on(mask, [(p[0], p[1]) for p in pts_float])
            if mask.any():
                ann_id = mgr.add_annotation(cat.id, mask)
                ann = mgr.get_annotation(ann_id)
                if ann is not None:
                    ann.original_polygons = [pts_float]

        mask_managers[img_ann.image_id] = mgr

    return project, mask_managers


def has_labelme_annotations(folder: str, image_filenames: List[str]) -> bool:
    """Return True if at least one matching .json file exists in folder."""
    return any(
        os.path.isfile(os.path.join(folder, os.path.splitext(f)[0] + ".json"))
        for f in image_filenames
    )
