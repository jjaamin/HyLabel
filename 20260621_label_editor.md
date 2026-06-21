# Label Editor 작업 노트 — 2026-06-21

## 프로젝트 개요

**경로**: `C:\Users\monG\label_editor\`  
**스택**: PyQt6 · NumPy · OpenCV · COCO JSON

이미지 라벨링 앱. 폴리곤/브러쉬로 mask를 그리고 Enter 키로 개별 annotation으로 커밋. COCO JSON 형식으로 저장.

---

## 파일 구조

```
label_editor/
├── main.py
└── labeler/
    ├── __init__.py
    ├── canvas.py        # QGraphicsView 기반 캔버스
    ├── mask_manager.py  # Annotation 저장·렌더링·컨투어 추출
    ├── window.py        # QMainWindow (UI·시그널·파일 I/O)
    ├── models.py        # Project, ImageAnnotation, Category 데이터 클래스
    └── coco_io.py       # COCO JSON 저장/로드
```

---

## 핵심 아키텍처

### Annotation 모델 (`mask_manager.py`)

```python
@dataclass
class Annotation:
    ann_id: int
    cat_id: int
    mask: np.ndarray   # H×W uint8

class MaskManager:
    _annotations: List[Annotation]
    _next_id: int
```

- 카테고리별 단일 마스크 대신 **개별 Annotation 리스트**로 관리
- `add_annotation(cat_id, mask) → ann_id`
- `remove_annotation(ann_id)`
- `to_coco_annotations(image_id)` / `load_from_coco(annotations)`

### Pending Mask 패턴 (`canvas.py`)

- 드로잉 중인 내용은 `_pending_mask`(임시 버퍼)에 기록
- **Enter 키**를 눌러야만 `MaskManager.add_annotation()`으로 커밋 → Labels 패널에 추가
- Esc: pending 버림

### Edit 모드

- Labels 패널에서 annotation 선택 → `set_edit_annotation(ann_id, mask)`
- 브러쉬/지우개: `_edit_mask`에 직접 기록 (pending 아님)
- 컨트롤 포인트 드래그로 폴리곤 shape 변경
- Esc: edit 모드 종료

---

## 주요 기능 구현 이력

### 1. 컨투어 표시 + 컨트롤 포인트 드래그

**목표**: label 선택 시 경계선과 드래그 가능한 yellow dot 표시.

**최종 구현**:
- **경계선 표시**: `MaskManager.boundary_rgba(mask)` — `mask & ~erode(mask)`로 경계 픽셀만 추출해 흰색 RGBA 렌더링. 벡터 경로 대신 픽셀 overlay(`_contour_overlay`, z=8)로 mask와 완벽 일치.
- **컨트롤 포인트**: `MaskManager.extract_cp_contours(mask)` — TC89_L1 컨투어 + **+0.5 shift**(픽셀 센터 좌표). 노란 `QGraphicsEllipseItem` (ItemIgnoresTransformations, z=35).
- **드래그 preview**: 드래그 시작 시 pixel overlay 숨기고 노란 점선 polygon(`_cp_path_items`) 표시. 드래그 종료 후 `_commit_contour_edit()` → `_show_contour()` 로 pixel overlay 복원.

**경계선 표시 방식 변천**:
1. `approxPolyDP` (1%) → QPainterPath: 너무 단순화, mask 밖으로 선 삐져나옴
2. `TC89_L1` → QPainterPath: 개선됐지만 벡터 경로 한계 (픽셀 좌상단 vs 센터 불일치)
3. `CHAIN_APPROX_NONE` + +0.5 shift → QPainterPath: 이론상 맞지만 실제로 여전히 부정확
4. **픽셀 overlay** (`boundary_rgba`): mask 경계 픽셀 그대로 렌더링 → 완벽 일치 ✓
5. 사용자 요청으로 경계선(흰색 1px) 제거 → **yellow dot만 표시**

**현재 표시**: 컨트롤 포인트 점(노란 dot)만 표시. 경계선 없음.

### 2. 컨트롤 포인트 드래그 → mask 재래스터화

```python
def _commit_contour_edit(self) -> None:
    self._edit_mask[:] = 0
    for pts in self._cp_contours:
        MaskManager.fill_polygon_on(self._edit_mask, pts)
    self._refresh_overlay_full()
    self.edit_changed.emit()
```

- 드래그 완료(mouseRelease) 후 `_cp_contours`(TC89_L1 pixel-center 좌표)로 mask 재래스터화
- `TC89_L1` 84pts 기준 commit 시 mask diff ≈ 80px (이전 `approxPolyDP` 4pts 대비 388px → 5배 개선)

### 3. 브러쉬 편집 후 컨투어 실시간 갱신

```python
# mouseReleaseEvent (BRUSH mode)
if self._edit_ann_id >= 0:
    self._show_contour()   # stroke 완료 후 dot 갱신
```

### 4. [ ] 키 브러쉬 크기 버그 수정

**원인**: 툴바 버튼 클릭 후 포커스가 툴바에 남아 canvas `keyPressEvent` 미수신.  
**수정**: `_on_tool_toggled`에서 모드 설정 후 `self.canvas.setFocus()` 호출.

### 5. Edit 모드 후 새 label 추가 버그 수정

**원인**: edit 모드(`_edit_ann_id >= 0`)에서 Enter가 무조건 차단.  
**수정 1**: Class list 클릭 시 edit 모드 해제.  
**수정 2**: DRAW 모드에서 draft 3점 이상이면 edit 모드 자동 해제 후 커밋.

```python
# canvas.py keyPressEvent
elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
    if self._edit_ann_id >= 0:
        if self._mode == Mode.DRAW and len(self._draft_pts) >= 3:
            self.clear_edit_annotation()
        else:
            return
```

### 6. Class / Label 굵은 글씨 표시

| 상황 | Classes | Labels |
|---|---|---|
| Class 선택 | **굵게** | 해제 |
| Label 선택 | 해제 | **굵게** |
| Esc (edit 종료) | — | 해제 |

- Class: `_update_class_bold()` (`currentRowChanged` 연결)
- Class 클릭: `_on_class_clicked()` → `clear_edit_annotation()` + `_clear_label_bold()`
- Label 선택: `_set_label_bold(row)` + `_clear_class_bold()`
- Edit 종료: `_on_edit_cleared()` → `_clear_label_bold()`

**주의**: `_clear_class_bold()`는 `blockSignals(True/False)`로 감싸서 bold 해제가 `_update_active_class` 연쇄 호출을 일으키지 않도록 처리.

**Class 클릭 시그널**: `clicked` 사용 (`currentRowChanged`는 같은 항목 재클릭 시 미발생).

---

## MaskManager 주요 정적 메서드

```python
# 페인트 (pending_mask 또는 edit_mask에 직접 적용)
MaskManager.paint_circle_on(mask, cx, cy, radius) → (x1, y1, x2, y2)
MaskManager.erase_circle_on(mask, cx, cy, radius) → (x1, y1, x2, y2)
MaskManager.fill_polygon_on(mask, points)         → (x1, y1, x2, y2)

# 컨투어
MaskManager.boundary_rgba(mask) → RGBA np.ndarray   # 경계 픽셀 흰색
MaskManager.extract_cp_contours(mask)               # TC89_L1 + 0.5 shift

# 렌더링
MaskManager.rgba_region(x1,y1,x2,y2, cat_colors, pending_mask, pending_cat_id)
MaskManager.full_rgba(cat_colors, pending_mask, pending_cat_id)
```

---

## Canvas 시그널

```python
annotation_committed = pyqtSignal()   # Enter → 새 annotation 커밋
stroke_finished      = pyqtSignal()   # 브러쉬 mouse release
mode_changed         = pyqtSignal(str) # "idle"|"pan"|"draw"|"brush"
brush_size_changed   = pyqtSignal(int)
edit_changed         = pyqtSignal()   # 브러쉬/포인트 드래그로 annotation 수정
edit_cleared         = pyqtSignal()   # edit 모드 종료
```

---

## Canvas 내부 상태

```python
_pending_mask: Optional[np.ndarray]   # 드로잉 중 임시 버퍼
_edit_ann_id: int                      # 편집 중 annotation ID (-1 = 없음)
_edit_mask: Optional[np.ndarray]       # annotation mask 직접 참조

_contour_overlay: _MaskOverlayItem     # 경계 픽셀 overlay (z=8, 현재 미사용)
_cp_contours: List[List[Tuple]]        # TC89_L1 컨트롤 포인트 (pixel center)
_cp_path_items: List[QGraphicsPathItem] # 드래그 preview polygon (기본 hidden)
_cp_dot_items: List[List[QGraphicsEllipseItem]]  # yellow dot
_dragging_cp: Tuple[int, int]          # (contour_idx, point_idx), -1,-1=없음
```

---

## 키 바인딩

| 키 | 기능 |
|---|---|
| D | Draw (폴리곤) 모드 |
| B | Brush 모드 |
| H | Pan 모드 |
| F | Fit view |
| Space (hold) | 임시 Pan |
| Enter | pending mask 커밋 (새 label 추가) |
| Esc | 취소 / edit 모드 종료 |
| [ / ] | 브러쉬 크기 -/+ |
| = / - | 줌 in/out |
| 휠 | 줌 |

---

## 알려진 동작 특성

- **컨트롤 포인트 드래그 후 mask diff**: TC89_L1 근사 특성상 이동 없이 commit해도 경계 픽셀 ~80px 차이 발생 (이전 approxPolyDP 대비 5배 개선됨)
- **브러쉬 edit 모드**: edit 모드에서 B 누르면 선택된 annotation에 직접 paint/erase. Class 먼저 클릭(→ edit 해제) 후 B 누르면 새 annotation에 paint.
- **경계선 overlay**: `boundary_rgba` / `_contour_overlay` 코드는 유지되나 현재 `_show_contour()`에서 호출 안 함 (사용자 요청으로 dot만 표시).
