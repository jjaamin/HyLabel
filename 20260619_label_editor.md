# Label Editor — 작업 진행 기록

## 프로젝트 개요

- **경로**: `C:\Users\monG\label_editor\`
- **실행**: `C:\Users\monG\AppData\Local\Programs\Python\Python313\python.exe run_labeler.py`
- **목적**: PyQt6 기반 이미지 라벨링 GUI 프로그램 (COCO JSON 출력)
- **Python**: 3.13 (`C:\Users\monG\AppData\Local\Programs\Python\Python313\`)

---

## 파일 구조

```
label_editor/
├── run_labeler.py          # 진입점 (labeler.main.main 호출)
├── requirements.txt        # PyQt6>=6.4.0, numpy>=1.24.0, opencv-python>=4.8.0
└── labeler/
    ├── __init__.py
    ├── main.py             # QApplication 생성, Fusion 스타일, MainWindow 실행
    ├── window.py           # MainWindow (UI 전체)
    ├── canvas.py           # ImageCanvas (QGraphicsView 기반 캔버스)
    ├── mask_manager.py     # MaskManager (numpy 마스크 저장/편집)
    ├── models.py           # 데이터 모델 (Category, Project 등)
    └── coco_io.py          # COCO JSON 저장/불러오기
```

---

## 구현된 주요 기능

### 도구 (좌측 세로 툴바)
| 버튼 | 단축키 | 기능 |
|------|--------|------|
| ⬠ (Draw Polygon) | D | 다각형 클릭으로 그리기, 더블클릭 or 첫 점 스냅으로 완성 |
| 🖌 (Brush) | B | LMB=브러쉬 칠하기 / RMB=지우기 (통합) |
| *(separator)* | | |
| 🖐 (Pan) | H | 클릭-드래그로 이미지 이동 |
| 🔍+ (Zoom In) | = | 확대 |
| 🔍- (Zoom Out) | - | 축소 |
| ⊡ (Fit Image) | F | 이미지를 뷰에 맞춤 |

- 마우스 휠: 줌인/줌아웃
- Space 키(홀드): 임시 Pan 모드
- `[` / `]`: 브러쉬 크기 -1 / +1 조절 (범위 1~50)

### 우측 패널 구성 (위→아래)
1. **Brush 크기 슬라이더** (1~50)
2. **Classes** — 클래스 추가/제거, 색상 자동 할당. 선택된 클래스 굵은 글씨 표시.
   - 추가 시 기본 이름 자동 입력: Class1, Class2, ... (기존 이름 중 비어있는 번호)
3. **Labels** — 클래스별 마스크 칠해진 여부 표시 (●/○), Clear Active Label 버튼
4. **Images** — 이미지 목록 (클릭으로 전환)

### 저장/불러오기
- **저장**: File → Save (Ctrl+S), Save As (Ctrl+Shift+S)
- **불러오기**: File → Load Annotations, 또는 폴더 열 때 `annotations.json` 자동 감지
- **형식**: COCO JSON (segmentation polygon 형식)
- 작업 중에는 numpy 마스크로 보관, 저장 시에만 `cv2.findContours`로 폴리곤 변환
- **최근 경로 기억**: `QSettings`로 마지막 열었던 폴더/파일 경로 저장 (재시작 후에도 유지)

---

## 핵심 기술 구조

### canvas.py

```
Mode Enum: IDLE / PAN / DRAW / BRUSH

ImageCanvas(QGraphicsView)
  ├── _MaskOverlayItem(QGraphicsItem)  ← numpy RGBA → QImage 빠른 렌더링
  ├── _draft_pts / _draft_path         ← 폴리곤 그리기 중 미리보기
  └── _brush_ring                      ← 브러쉬 크기 커서 (흰 점선 원)

주요 필드:
  _stroke_erase: bool   브러쉬 모드에서 RMB 클릭 시 True (지우개 동작)

Signals:
  mask_changed()        마스크 변경 시
  stroke_finished()     브러쉬 stroke 완료 시
  mode_changed(str)     모드 변경 시 ("idle"/"pan"/"draw"/"brush")
  brush_size_changed(int)
```

**BRUSH 모드 동작**:
- LMB drag → `paint_circle` (현재 카테고리에 칠함)
- RMB drag → `erase_circle` (모든 카테고리에서 지움)
- 우클릭 컨텍스트 메뉴 비활성화 (`NoContextMenu`)

**성능 최적화**: 브러쉬 stroke마다 dirty rect만 업데이트 (`QGraphicsItem.update(rect)`).
전체 이미지 재처리 없이 변경된 영역만 numpy→QImage 변환.

### mask_manager.py

```python
class MaskManager:
    # 카테고리별 numpy uint8 마스크 보관 (overlap 허용)
    _masks: Dict[int, np.ndarray]  # cat_id → H×W 마스크

    paint_circle(cat_id, cx, cy, radius)  # cv2.circle
    erase_circle(cx, cy, radius)          # 모든 카테고리에서 지우기
    fill_polygon(cat_id, points)          # cv2.fillPoly
    rgba_region(x1,y1,x2,y2, cat_colors) # dirty rect → RGBA numpy (overlap=평균색)
    full_rgba(cat_colors)                 # 전체 → RGBA numpy
    to_coco_annotations(image_id)         # 저장 시만: findContours → polygon
    load_from_coco(annotations)           # COCO polygon → 마스크 래스터화
```

**Overlap 렌더링**: `rgba_region`에서 같은 픽셀에 여러 카테고리가 겹치면
각 카테고리 색의 평균값으로 표시 (float 누적 → count로 나눔).

### window.py 아이콘 생성 함수

```python
_color_icon(hex_color, size=14)   # 클래스 색상 정사각형 아이콘
_emoji_icon(symbol, size=21)      # 이모지 → QPixmap (Brush, Pan 버튼)
_zoom_icon(plus, size=21)         # QPainter로 직접 그린 돋보기+/- 아이콘
_fit_icon(size=21)                # QPainter로 직접 그린 사진+대각선화살표 아이콘
_polygon_icon(size=21)            # QPainter로 직접 그린 오각형+꼭짓점 점 아이콘
```

- 아이콘 내부 좌표 전부 `s = float(size)` 비례 설계 (어떤 크기에서도 정상 렌더링)
- 툴바 `setIconSize(QSize(21, 21))`, 버튼 min 29×29px, padding 4px

---

## 수정 이력

### 버그 수정: Draw Polygon 완성 후 새 선 시작 문제

**수정 내용** (`canvas.py`):
1. `_complete_draw()` 끝에 `self.set_mode(Mode.IDLE)` → 완성 후 자동 IDLE 복귀
2. `mouseDoubleClickEvent`에 `else: self._cancel_draw()` → orphan 아이템 정리

### 20260619 세션 변경 사항

| 항목 | 내용 |
|------|------|
| 우측 패널 명칭 | Layers → **Labels**, 헤더 bold 제거 |
| 변수 이름 | `_layer_list` → `_label_list`, `_refresh_layers` → `_refresh_labels`, `_clear_active_layer` → `_clear_active_label`, `_btn_clear_layer` → `_btn_clear_label` |
| 브러쉬 크기 | 범위 1~200 → **1~50**, 단축키 간격 ±5 → **±1** |
| Brush/Eraser 통합 | Mode.ERASER 제거. BRUSH 모드에서 LMB=칠하기, RMB=지우기 |
| 클래스 기본 이름 | 추가 다이얼로그에 **Class1, Class2...** 자동 입력 |
| 최근 경로 기억 | `QSettings("LabelEditor","LabelEditor")` key=`lastDir` |
| Classes 선택 강조 | 선택된 클래스 항목 **bold** 표시 (`currentRowChanged` 연결) |
| Overlap 지원 | 겹치는 픽셀을 카테고리 색 평균으로 렌더링 |
| 툴바 아이콘 축소 | 40px → 21px (≈1/3 후 1.5배), 내부 좌표 size 비례 재설계 |
| 툴바 구성 변경 | Eraser 버튼 제거, Hand→**Pan** 명칭 통일, separator를 Brush↔Pan 사이로 이동 |

---

## 환경 메모

- PowerShell에서 `&&` 연산자 미지원 → `;` 또는 개별 명령 사용
- 패키지 설치: `C:\Users\monG\AppData\Local\Programs\Python\Python313\python.exe -m pip install <패키지>`
- 파일 인코딩: `open(file, encoding='utf-8')` 명시 필요 (이모지 포함)
- 설치된 패키지: PyQt6, numpy-2.4.6, opencv-python-4.13.0.92

---

## 다음에 할 수 있는 작업 (미결)

- 특별히 요청된 미완성 기능 없음
- 필요 시 추가 기능 논의 후 진행
