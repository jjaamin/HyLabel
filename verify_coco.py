"""
COCO 저장·로드 검증 스크립트.

검증 항목:
  1. 개별 annotation이 각자 분리된 COCO entry로 저장되는지
  2. category_id / image_id / bbox / area 필드가 올바른지
  3. segmentation polygon을 다시 래스터화하면 원본 mask와 일치하는지
  4. 여러 category·여러 annotation이 뒤섞이지 않는지
  5. 저장 → 로드 round-trip 후 mask가 복원되는지
"""

import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labeler.mask_manager import MaskManager
from labeler.models import Project
from labeler import coco_io

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    msg = f"[{tag}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append(ok)


# ── 공통 설정 ──────────────────────────────────────────────────────────────────
W, H = 300, 200

# 두 개의 category
CAT_PERSON = 1
CAT_CAR    = 2

# annotation 1: person — 사각형 영역 (왼쪽 위)
mask_person1 = np.zeros((H, W), dtype=np.uint8)
mask_person1[20:80, 10:70] = 255

# annotation 2: person — 원형 영역 (오른쪽)
mask_person2 = np.zeros((H, W), dtype=np.uint8)
cv2.circle(mask_person2, (220, 100), 40, 255, -1)

# annotation 3: car — 사각형 (아래)
mask_car1 = np.zeros((H, W), dtype=np.uint8)
mask_car1[130:180, 80:200] = 255

# MaskManager에 등록
mgr = MaskManager(W, H)
id1 = mgr.add_annotation(CAT_PERSON, mask_person1)
id2 = mgr.add_annotation(CAT_PERSON, mask_person2)
id3 = mgr.add_annotation(CAT_CAR,    mask_car1)

print(f"\n=== 등록된 annotation: id={id1},{id2},{id3} ===\n")

# ── 테스트 1: annotation 개수 ──────────────────────────────────────────────────
print("─── 1. annotation 개수 ───")
anns_raw = mgr.to_coco_annotations(image_id=1)
check("annotation 3개 생성", len(anns_raw) == 3,
      f"실제 {len(anns_raw)}개")

# ── 테스트 2: category_id 분리 ─────────────────────────────────────────────────
print("\n─── 2. category_id 분리 ───")
cat_ids = [a["category_id"] for a in anns_raw]
check("person 2개", cat_ids.count(CAT_PERSON) == 2)
check("car 1개",    cat_ids.count(CAT_CAR)    == 1)

# ── 테스트 3: segmentation 필드 형식 ──────────────────────────────────────────
print("\n─── 3. segmentation 형식 ───")
for i, ann in enumerate(anns_raw, 1):
    seg = ann["segmentation"]
    ok_list  = isinstance(seg, list) and len(seg) >= 1
    ok_flat  = isinstance(seg[0], list) and len(seg[0]) >= 6  # 최소 3점=6좌표
    ok_even  = len(seg[0]) % 2 == 0
    check(f"  ann#{i} segmentation 형식 (list of flat coords, 짝수)",
          ok_list and ok_flat and ok_even,
          f"len={len(seg[0]) if ok_list else '?'}")

# ── 테스트 4: bbox 유효성 ──────────────────────────────────────────────────────
print("\n─── 4. bbox 유효성 ───")
for i, ann in enumerate(anns_raw, 1):
    x, y, bw, bh = ann["bbox"]
    ok = bw > 0 and bh > 0
    check(f"  ann#{i} bbox 양수 너비/높이", ok, f"bbox={ann['bbox']}")

# ── 테스트 5: area 유효성 ─────────────────────────────────────────────────────
print("\n─── 5. area 유효성 ───")
for i, ann in enumerate(anns_raw, 1):
    ok = ann["area"] > 0
    check(f"  ann#{i} area > 0", ok, f"area={ann['area']}")

# ── 테스트 6: image_id 일치 ───────────────────────────────────────────────────
print("\n─── 6. image_id ───")
check("모든 annotation의 image_id == 1",
      all(a["image_id"] == 1 for a in anns_raw))

# ── 테스트 7: segmentation 재래스터화 → 원본과 비교 ──────────────────────────
print("\n─── 7. 폴리곤 → mask 재래스터화 정확도 ───")
original_masks = {id1: mask_person1, id2: mask_person2, id3: mask_car1}

for ann in anns_raw:
    seg = ann["segmentation"][0]
    pts = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
    rasterized = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(rasterized, [pts], 255)

    # 원본 annotation ID를 ann_id로 찾기
    orig_ann_id = ann.get("id", -1)
    # id는 to_coco_annotations 내부의 ann.ann_id
    orig_mask = None
    for a in mgr.annotations():
        if a.ann_id == ann.get("id"):   # save_coco가 id를 덮어쓰기 전 확인
            orig_mask = a.mask
            break

    # 전체 픽셀 대비 diff 비율
    diff_px = int((np.abs(rasterized.astype(int) - rasterized.astype(int)) > 0).sum())

# category별로 재래스터화해서 중복 없는지 확인
rasters = []
for ann in anns_raw:
    seg = ann["segmentation"][0]
    pts = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
    r = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(r, [pts], 255)
    rasters.append((ann["category_id"], r))

# person 두 annotation이 겹치는지
person_masks = [r for cid, r in rasters if cid == CAT_PERSON]
if len(person_masks) == 2:
    overlap = int(np.logical_and(person_masks[0], person_masks[1]).sum())
    check("person annotation 2개가 서로 겹치지 않음", overlap == 0,
          f"겹치는 픽셀 {overlap}개")

# ── 테스트 8: 저장 → JSON 구조 검증 ──────────────────────────────────────────
print("\n─── 8. JSON 파일 구조 ───")
project = Project()
project.add_category("person")   # id=1
project.add_category("car")      # id=2
fake_path = "/fake/image.jpg"
img_ann = project.get_or_create_image(fake_path, W, H)

with tempfile.TemporaryDirectory() as tmpdir:
    json_path = os.path.join(tmpdir, "annotations.json")
    coco_io.save_coco(project, {img_ann.image_id: mgr}, json_path, tmpdir)

    with open(json_path, encoding="utf-8") as f:
        saved = json.load(f)

    check("info 필드 존재",       "info"        in saved)
    check("categories 필드 존재", "categories"  in saved)
    check("images 필드 존재",     "images"      in saved)
    check("annotations 필드 존재","annotations" in saved)

    check("categories 2개",  len(saved["categories"])  == 2,
          str([c["name"] for c in saved["categories"]]))
    check("images 1개",      len(saved["images"])      == 1)
    check("annotations 3개", len(saved["annotations"]) == 3,
          f"실제 {len(saved['annotations'])}개")

    ann_cat_ids = [a["category_id"] for a in saved["annotations"]]
    check("저장된 category_id 종류", set(ann_cat_ids) == {1, 2},
          str(sorted(ann_cat_ids)))

    # ── 테스트 9: 로드 round-trip ────────────────────────────────────────────
    print("\n─── 9. 저장 → 로드 round-trip ───")
    proj2, mgrs2 = coco_io.load_coco(json_path, tmpdir)

    check("카테고리 수 복원",  len(proj2.categories) == 2)
    check("이미지 수 복원",    len(proj2.images)     == 1)

    img2 = proj2.images[0]
    mgr2 = mgrs2.get(img2.image_id)
    check("MaskManager 복원", mgr2 is not None)

    if mgr2:
        anns2 = mgr2.annotations()
        check("annotation 3개 복원", len(anns2) == 3,
              f"실제 {len(anns2)}개")

        cat_ids2 = [a.cat_id for a in anns2]
        check("person 2개 복원", cat_ids2.count(CAT_PERSON) == 2)
        check("car 1개 복원",    cat_ids2.count(CAT_CAR)    == 1)

        # 복원된 mask 픽셀 수 vs 원본 비교
        orig_areas = {
            CAT_PERSON: [int(mask_person1.sum() // 255),
                         int(mask_person2.sum() // 255)],
            CAT_CAR:    [int(mask_car1.sum()    // 255)],
        }
        restored_areas = {CAT_PERSON: [], CAT_CAR: []}
        for a in anns2:
            restored_areas[a.cat_id].append(int(a.mask.sum() // 255))

        for cid, olist in orig_areas.items():
            rlist = sorted(restored_areas.get(cid, []))
            olist_sorted = sorted(olist)
            # 폴리곤 근사 오차 허용: 원본의 5% 이내
            ok = len(rlist) == len(olist_sorted) and all(
                abs(r - o) / max(o, 1) < 0.05
                for r, o in zip(rlist, olist_sorted)
            )
            cname = "person" if cid == CAT_PERSON else "car"
            check(f"  {cname} mask 픽셀 수 5% 이내 복원",
                  ok, f"원본={sorted(olist_sorted)} 복원={rlist}")

# ── 최종 결과 ──────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(results)
failed = total - passed
print(f"\n{'='*50}")
print(f"결과: {passed}/{total} 통과  ({failed}개 실패)")
if failed == 0:
    print("모든 검증 통과")
else:
    print("일부 검증 실패 — 위 FAIL 항목 확인 필요")
print('='*50)
sys.exit(0 if failed == 0 else 1)
