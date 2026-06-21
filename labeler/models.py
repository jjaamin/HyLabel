from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

PALETTE = [
    "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
    "#911EB4", "#42D4F4", "#F032E6", "#BFEF45", "#FABEBE",
    "#469990", "#E6BEFF", "#9A6324", "#FFFAC8", "#800000",
    "#AAFFC3", "#808000", "#FFD8B1", "#000075", "#A9A9A9",
]


@dataclass
class Category:
    id: int
    name: str
    color: str
    supercategory: str = ""


@dataclass
class AnnotationPolygon:
    id: int
    category_id: int
    points: List[Tuple[float, float]]

    def to_coco_segmentation(self) -> list:
        return [[coord for x, y in self.points for coord in (x, y)]]

    def bbox(self) -> list:
        if not self.points:
            return [0.0, 0.0, 0.0, 0.0]
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x, y = min(xs), min(ys)
        return [x, y, max(xs) - x, max(ys) - y]

    def area(self) -> float:
        n = len(self.points)
        if n < 3:
            return 0.0
        s = 0.0
        for i in range(n):
            j = (i + 1) % n
            s += self.points[i][0] * self.points[j][1]
            s -= self.points[j][0] * self.points[i][1]
        return abs(s) / 2.0


@dataclass
class ImageAnnotation:
    image_id: int
    file_path: str
    width: int
    height: int
    polygons: List[AnnotationPolygon] = field(default_factory=list)
    _next_ann_id: int = field(default=1, repr=False)

    def add_polygon(self, ann_id: int, category_id: int,
                    points: List[Tuple[float, float]]) -> AnnotationPolygon:
        ann = AnnotationPolygon(id=ann_id, category_id=category_id, points=points)
        self.polygons.append(ann)
        self._next_ann_id = max(self._next_ann_id, ann_id + 1)
        return ann

    def remove_polygon(self, ann_id: int) -> None:
        self.polygons = [p for p in self.polygons if p.id != ann_id]

    def update_polygon_points(self, ann_id: int,
                               points: List[Tuple[float, float]]) -> None:
        for p in self.polygons:
            if p.id == ann_id:
                p.points = points
                break


@dataclass
class Project:
    categories: List[Category] = field(default_factory=list)
    images: List[ImageAnnotation] = field(default_factory=list)
    _next_cat_id: int = field(default=1, repr=False)
    _next_img_id: int = field(default=1, repr=False)

    def add_category(self, name: str, supercategory: str = "") -> Category:
        color = PALETTE[(self._next_cat_id - 1) % len(PALETTE)]
        cat = Category(id=self._next_cat_id, name=name, color=color,
                       supercategory=supercategory)
        self._next_cat_id += 1
        self.categories.append(cat)
        return cat

    def get_category(self, cat_id: int) -> Optional[Category]:
        return next((c for c in self.categories if c.id == cat_id), None)

    def get_or_create_image(self, file_path: str, width: int,
                             height: int) -> ImageAnnotation:
        for img in self.images:
            if img.file_path == file_path:
                return img
        img = ImageAnnotation(
            image_id=self._next_img_id,
            file_path=file_path,
            width=width,
            height=height,
        )
        self._next_img_id += 1
        self.images.append(img)
        return img
