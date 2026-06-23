from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Annotation:
    ann_id: int
    cat_id: int
    mask: np.ndarray  # H×W uint8


class MaskManager:
    """
    Per-image annotation storage.
    Each committed annotation is a separate (ann_id, cat_id, mask) entry.
    Pending mask lives in canvas until Enter commits it here.
    """

    ALPHA = 150

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._annotations: List[Annotation] = []
        self._next_id: int = 1

    # ── annotation management ─────────────────────────────────────────────────

    def add_annotation(self, cat_id: int, mask: np.ndarray) -> int:
        """Commit a mask as a new annotation. Returns the new ann_id."""
        ann_id = self._next_id
        self._next_id += 1
        self._annotations.append(Annotation(ann_id, cat_id, mask.copy()))
        return ann_id

    def remove_annotation(self, ann_id: int) -> None:
        self._annotations = [a for a in self._annotations if a.ann_id != ann_id]

    def get_annotation(self, ann_id: int) -> Optional[Annotation]:
        for ann in self._annotations:
            if ann.ann_id == ann_id:
                return ann
        return None

    def annotation_index(self, ann_id: int) -> int:
        for i, ann in enumerate(self._annotations):
            if ann.ann_id == ann_id:
                return i
        return -1

    def restore_annotation(self, ann_id: int, cat_id: int,
                           mask: np.ndarray, index: Optional[int] = None) -> None:
        """Re-insert a previously removed annotation at its original position."""
        self._annotations = [a for a in self._annotations if a.ann_id != ann_id]
        ann = Annotation(ann_id, cat_id, mask.copy())
        if index is not None:
            idx = min(index, len(self._annotations))
            self._annotations.insert(idx, ann)
        else:
            self._annotations.append(ann)
        self._next_id = max(self._next_id, ann_id + 1)

    def change_annotation_category(self, ann_id: int, new_cat_id: int) -> bool:
        for ann in self._annotations:
            if ann.ann_id == ann_id:
                ann.cat_id = new_cat_id
                return True
        return False

    def sort_by_category_order(self, cat_order: List[int]) -> None:
        order_map = {cat_id: i for i, cat_id in enumerate(cat_order)}
        self._annotations.sort(key=lambda a: order_map.get(a.cat_id, len(cat_order)))

    def annotations(self) -> List[Annotation]:
        return list(self._annotations)

    def has_any(self, cat_id: int) -> bool:
        return any(a.cat_id == cat_id and a.mask.any() for a in self._annotations)

    def active_categories(self) -> List[int]:
        seen: set = set()
        result = []
        for a in self._annotations:
            if a.mask.any() and a.cat_id not in seen:
                seen.add(a.cat_id)
                result.append(a.cat_id)
        return result

    def clear_category(self, cat_id: int) -> None:
        """Remove all annotations for a given category."""
        self._annotations = [a for a in self._annotations if a.cat_id != cat_id]

    # ── static paint helpers (operate on an external numpy mask) ─────────────

    @staticmethod
    def paint_circle_on(mask: np.ndarray, cx: int, cy: int,
                        radius: int) -> Tuple[int, int, int, int]:
        h, w = mask.shape
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        return (max(0, cx - radius - 1), max(0, cy - radius - 1),
                min(w, cx + radius + 2), min(h, cy + radius + 2))

    @staticmethod
    def erase_circle_on(mask: np.ndarray, cx: int, cy: int,
                        radius: int) -> Tuple[int, int, int, int]:
        h, w = mask.shape
        cv2.circle(mask, (cx, cy), radius, 0, -1)
        return (max(0, cx - radius - 1), max(0, cy - radius - 1),
                min(w, cx + radius + 2), min(h, cy + radius + 2))

    @staticmethod
    def fill_polygon_on(mask: np.ndarray,
                        points: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        h, w = mask.shape
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (max(0, int(min(xs)) - 1), max(0, int(min(ys)) - 1),
                min(w, int(max(xs)) + 2), min(h, int(max(ys)) + 2))

    # ── overlay rendering ─────────────────────────────────────────────────────

    def rgba_region(self, x1: int, y1: int, x2: int, y2: int,
                    cat_colors: Dict[int, Tuple[int, int, int]],
                    pending_mask: Optional[np.ndarray] = None,
                    pending_cat_id: int = -1) -> np.ndarray:
        """RGBA composite of committed annotations + optional pending mask."""
        h, w = y2 - y1, x2 - x1
        acc = np.zeros((h, w, 3), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.uint8)

        for ann in self._annotations:
            region = ann.mask[y1:y2, x1:x2]
            if not region.any():
                continue
            hit = region > 0
            r, g, b = cat_colors.get(ann.cat_id, (255, 0, 0))
            acc[hit, 0] += r
            acc[hit, 1] += g
            acc[hit, 2] += b
            count[hit] += 1

        if pending_mask is not None and pending_cat_id >= 0 and pending_mask.any():
            hit = pending_mask > 0
            r, g, b = cat_colors.get(pending_cat_id, (255, 0, 0))
            acc[hit, 0] += r
            acc[hit, 1] += g
            acc[hit, 2] += b
            count[hit] += 1

        out = np.zeros((h, w, 4), dtype=np.uint8)
        any_hit = count > 0
        if any_hit.any():
            c = count[any_hit, np.newaxis].astype(np.float32)
            out[any_hit, :3] = np.clip(acc[any_hit] / c, 0, 255).astype(np.uint8)
            out[any_hit, 3] = self.ALPHA
        return out

    def full_rgba(self, cat_colors: Dict[int, Tuple[int, int, int]],
                  pending_mask: Optional[np.ndarray] = None,
                  pending_cat_id: int = -1) -> np.ndarray:
        return self.rgba_region(0, 0, self.width, self.height, cat_colors,
                                pending_mask, pending_cat_id)

    # ── contour / boundary helpers ────────────────────────────────────────────

    @staticmethod
    def boundary_rgba(mask: np.ndarray) -> np.ndarray:
        """RGBA image with boundary pixels drawn white — pixel-perfect outline."""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = mask & ~eroded
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[boundary > 0] = [255, 255, 255, 230]
        return rgba

    @staticmethod
    def extract_cp_contours(mask: np.ndarray) -> List[List[Tuple[float, float]]]:
        """TC89_L1 contour points at pixel centres (+0.5) for control-point dragging."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        result = []
        for c in contours:
            if len(c) < 3:
                continue
            result.append([(float(p[0][0]) + 0.5, float(p[0][1]) + 0.5) for p in c])
        return result

    # ── COCO I/O ──────────────────────────────────────────────────────────────

    def to_coco_annotations(self, image_id: int) -> List[dict]:
        results = []
        for ann in self._annotations:
            if not ann.mask.any():
                continue
            contours, _ = cv2.findContours(
                ann.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
            )
            for c in contours:
                if len(c) < 3:
                    continue
                arc = cv2.arcLength(c, True)
                eps = max(1.0, 0.003 * arc)
                approx = cv2.approxPolyDP(c, eps, True)
                if len(approx) < 3:
                    continue
                flat = approx.reshape(-1).tolist()
                xs, ys = flat[0::2], flat[1::2]
                bx, by = min(xs), min(ys)
                results.append({
                    "id": ann.ann_id,
                    "image_id": image_id,
                    "category_id": ann.cat_id,
                    "segmentation": [flat],
                    "bbox": [bx, by, max(xs) - bx, max(ys) - by],
                    "area": round(float(cv2.contourArea(c)), 2),
                    "iscrowd": 0,
                })
        return results

    def load_from_coco(self, annotations: List[dict]) -> None:
        for ann in annotations:
            cat_id = ann["category_id"]
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            for seg in ann.get("segmentation", []):
                pts = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts], 255)
            if mask.any():
                ann_id = ann.get("id", self._next_id)
                self._next_id = max(self._next_id, ann_id + 1)
                self._annotations.append(Annotation(ann_id, cat_id, mask))
