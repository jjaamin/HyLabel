from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Annotation:
    ann_id: int
    cat_id: int
    mask: np.ndarray  # H×W uint8
    # Original polygon points loaded from LabelMe [[x,y], ...].
    # Preserved on save unless the mask was edited (then set to None).
    original_polygons: Optional[List] = field(default=None, repr=False, compare=False)
    # Tight bounds of the set pixels as (x1, y1, x2, y2), x2/y2 exclusive.
    # None means the mask is empty. Used to cull work in rgba_region().
    #
    # INVARIANT: bbox must never be smaller than the true extent, or those
    # pixels stop rendering. A too-large bbox only costs a little extra work,
    # so growing on paint and recomputing on shrink is always safe.
    bbox: Optional[Tuple[int, int, int, int]] = field(
        default=None, repr=False, compare=False)


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

    # ── bbox helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Tight (x1, y1, x2, y2) of set pixels, x2/y2 exclusive. None if empty."""
        rows = np.any(mask, axis=1)
        if not rows.any():
            return None
        cols = np.any(mask, axis=0)
        y1 = int(np.argmax(rows))
        y2 = int(len(rows) - np.argmax(rows[::-1]))
        x1 = int(np.argmax(cols))
        x2 = int(len(cols) - np.argmax(cols[::-1]))
        return (x1, y1, x2, y2)

    @staticmethod
    def union_bbox(a: Optional[Tuple[int, int, int, int]],
                   b: Optional[Tuple[int, int, int, int]]
                   ) -> Optional[Tuple[int, int, int, int]]:
        """Smallest rect covering both, ignoring None operands."""
        if a is None:
            return b
        if b is None:
            return a
        return (min(a[0], b[0]), min(a[1], b[1]),
                max(a[2], b[2]), max(a[3], b[3]))

    def recompute_bbox(self, ann_id: int) -> None:
        """Full recompute — call after an edit that may have shrunk the mask."""
        ann = self.get_annotation(ann_id)
        if ann is not None:
            ann.bbox = self.compute_bbox(ann.mask)

    def grow_bbox(self, ann_id: int, rect: Tuple[int, int, int, int]) -> None:
        """Union rect into the annotation bbox — call after painting into it."""
        ann = self.get_annotation(ann_id)
        if ann is None:
            return
        x1, y1, x2, y2 = rect
        if x2 <= x1 or y2 <= y1:
            return
        if ann.bbox is None:
            ann.bbox = (x1, y1, x2, y2)
        else:
            bx1, by1, bx2, by2 = ann.bbox
            ann.bbox = (min(bx1, x1), min(by1, y1), max(bx2, x2), max(by2, y2))

    # ── annotation management ─────────────────────────────────────────────────

    def add_annotation(self, cat_id: int, mask: np.ndarray) -> int:
        """Commit a mask as a new annotation. Returns the new ann_id."""
        ann_id = self._next_id
        self._next_id += 1
        m = mask.copy()
        self._annotations.append(
            Annotation(ann_id, cat_id, m, bbox=self.compute_bbox(m)))
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
                           mask: np.ndarray, index: Optional[int] = None,
                           original_polygons: Optional[List] = None) -> None:
        """Re-insert a previously removed annotation at its original position.

        original_polygons must be passed back too — dropping it would silently
        downgrade an authored polygon to a mask-extracted outline on save.
        """
        self._annotations = [a for a in self._annotations if a.ann_id != ann_id]
        m = mask.copy()
        ann = Annotation(ann_id, cat_id, m, original_polygons=original_polygons,
                         bbox=self.compute_bbox(m))
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

    @staticmethod
    def _blend_into(acc: np.ndarray, count: np.ndarray, hit: np.ndarray,
                    r: int, g: int, b: int) -> None:
        """Accumulate one colour into acc/count wherever hit is True, in place."""
        np.add(acc[:, :, 0], r, out=acc[:, :, 0], where=hit)
        np.add(acc[:, :, 1], g, out=acc[:, :, 1], where=hit)
        np.add(acc[:, :, 2], b, out=acc[:, :, 2], where=hit)
        np.add(count, 1, out=count, where=hit)

    def rgba_region(self, x1: int, y1: int, x2: int, y2: int,
                    cat_colors: Dict[int, Tuple[int, int, int]],
                    pending_mask: Optional[np.ndarray] = None,
                    pending_cat_id: int = -1) -> np.ndarray:
        """RGBA composite of committed annotations + optional pending mask."""
        h, w = y2 - y1, x2 - x1
        acc = np.zeros((h, w, 3), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.uint8)

        for ann in self._annotations:
            bb = ann.bbox
            if bb is None:
                continue
            bx1, by1, bx2, by2 = bb
            # Reject annotations whose bounds miss the requested region entirely.
            # This is what keeps the cost proportional to what is actually
            # visible rather than to (annotation count × image area).
            if bx2 <= x1 or bx1 >= x2 or by2 <= y1 or by1 >= y2:
                continue
            # Work only on the intersection of the bbox and the region.
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)

            hit = ann.mask[iy1:iy2, ix1:ix2] > 0
            if not hit.any():
                continue
            r, g, b = cat_colors.get(ann.cat_id, (255, 0, 0))
            # Views into acc/count — np.add(where=) writes through in place.
            # This is ~3x faster than boolean fancy indexing (`a[hit] += r`),
            # which allocates a gather buffer on every call.
            sub_acc = acc[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1]
            self._blend_into(sub_acc,
                             count[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1],
                             hit, r, g, b)

        if pending_mask is not None and pending_cat_id >= 0 and pending_mask.any():
            hit = pending_mask > 0
            r, g, b = cat_colors.get(pending_cat_id, (255, 0, 0))
            self._blend_into(acc, count, hit, r, g, b)

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
        """Full-density contour points at pixel centres (+0.5) for control-point
        dragging.

        CHAIN_APPROX_NONE is mandatory here: these points are not display-only —
        _commit_contour_edit() re-rasterizes the mask from them and writes them
        to Annotation.original_polygons. Any approximation (TC89_L1, approxPolyDP)
        would silently discard boundary detail on every edit.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        result = []
        for c in contours:
            if len(c) < 3:
                continue
            result.append([(float(p[0][0]) + 0.5, float(p[0][1]) + 0.5) for p in c])
        return result

    # ── LabelMe I/O ───────────────────────────────────────────────────────────

    def to_coco_annotations(self, image_id: int) -> List[dict]:
        results = []
        for ann in self._annotations:
            if not ann.mask.any():
                continue
            contours, _ = cv2.findContours(
                ann.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for c in contours:
                if len(c) < 3:
                    continue
                flat = c.reshape(-1).tolist()
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
                self._annotations.append(
                    Annotation(ann_id, cat_id, mask,
                               bbox=self.compute_bbox(mask)))
