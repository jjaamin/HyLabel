# HyLabel

이미지 어노테이션 도구 (LabelMe Format)

---

DMI 영상응용계측기술팀 구자민 TL

---

## 설치 (다른 PC에 처음 설치하는 경우)

### 1. Python 설치

Python 3.10 이상 필요 (개발은 3.13 기준). [python.org](https://www.python.org/downloads/) 에서 설치 시 **"Add python.exe to PATH"** 체크.

설치 확인:
```bash
python --version
```

### 2. 소스 받기

Git이 설치되어 있으면:
```bash
git clone https://github.com/jjaamin/HyLabel.git
cd HyLabel
```
Git이 없으면 GitHub에서 "Code → Download ZIP"으로 받아 압축 해제.

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`에 포함된 것: PyQt6, numpy, opencv-python, scipy, huggingface_hub, onnxruntime.

> `pip`이 없거나 오류가 나면 `python -m pip install --upgrade pip` 먼저 실행.

### 4. AI 매직완드 모델 다운로드 (선택)

AI 매직완드 기능은 **EdgeSAM**과 **SAM2**(Hiera-Tiny / Base+ / Large) 중 골라서 사용할 수 있습니다 (사이드바 "AI Model" 드롭다운). 사용할 모델의 가중치를 받아야 합니다. (이 기능을 안 쓸 거면 건너뛰어도 앱은 정상 실행됨)

SAM2는 크기가 클수록 배경 오검출이 줄고 정확해지지만 느려지고 가중치도 커집니다. Hiera-Tiny에서 배경이 같이 잡히는 경우 Base+ 또는 Large를 시도해보세요.

```bash
python download_weights.py                       # EdgeSAM (기본값)
python download_weights.py --model sam2           # SAM2 (Hiera-Tiny, ~134MB)
python download_weights.py --model sam2_base_plus # SAM2 (Hiera-Base+, ~340MB)
python download_weights.py --model sam2_large     # SAM2 (Hiera-Large, ~889MB)
python download_weights.py --model all            # 전부
```

**GPU 가속 사용 시** (NVIDIA GPU + CUDA 12.x):
```bash
pip install onnxruntime-gpu==1.20.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
onnxruntime-gpu는 cuDNN 9가 필요한데, 위처럼 PyTorch(CUDA 빌드)를 같이 설치해두면 앱이 PyTorch에 번들된 `cudnn64_9.dll`을 자동으로 찾아 쓰므로 cuDNN을 별도로 설치할 필요가 없습니다. PyTorch를 설치하지 않으면 cuDNN을 직접 설치해야 GPU 가속이 동작합니다.

**CPU 전용 (GPU 없는 회사 PC 등):**
```bash
pip install onnxruntime
```
그냥 `requirements.txt`에 이미 포함되어 있으므로 2번 단계로 충분하며, 별도 조치 불필요.

### 5. 설치 확인

```bash
python run_hylabel.py
```
창이 뜨면 설치 완료.

---

## 실행

```bash
python run_hylabel.py
```

---

## 화면 구성

```
┌─────────────────────────────────┬──────────────────┐
│  좌측 툴바                        │  우측 패널       │
│                                  │  - Brush 크기    │
│  캔버스 (이미지 + 마스크 오버레이)   │  - Mask 선택    │
│                                  │  - Classes       │
│                                  │  - Labels        │
│                                  │  - Images 목록   │
└─────────────────────────────────┴──────────────────┘
```

---

## 단축키

| 키 | 기능 |
|---|---|
| `D` | Draw 폴리곤 도구 |
| `B` | Brush 도구 |
| `M` | AI 매직완드 도구 |
| `H` | Pan (이동) / 한 번 더 누르면 이전 도구로 복귀 |
| `F` | 이미지 맞춤 (Fit) |
| `=` | 확대 |
| `-` | 축소 |
| `[` | 브러시 크기 줄이기 / 매직완드 마스크 선택 |
| `]` | 브러시 크기 키우기 / 매직완드 마스크 선택 |
| `Enter` | 마스크 확정 |
| `Esc` | 취소 / 편집 모드 종료 |
| `Ctrl+Z` | 실행 취소 |
| `Ctrl+S` | 저장 |
| `Ctrl+O` | 폴더 열기 |
| `Space` (누르는 동안) | 임시 Pan 모드 |
| `V` | 레이블 흐리게 보기 토글 |
| `G` | 감마 보정 토글 |

---

## 기능 설명

### 파일 메뉴

| 메뉴 | 설명 |
|---|---|
| Open Folder | 이미지 폴더 열기. 같은 폴더에 JSON이 있으면 자동 로드 |
| Open Image | 이미지 단일 파일 열기 |
| Load from Folder | 다른 폴더의 JSON 어노테이션 불러오기 |
| Save (`Ctrl+S`) | 현재 경로에 저장 |
| Save to Folder | 저장 경로 선택 후 저장 |

> 수정 사항이 있을 때 닫거나 새 폴더를 열면 **Save / Discard / Cancel** 창이 뜹니다.

---

### 이미지 목록

- 우측 하단에 이미지 파일 목록 표시
- **📄 파란 문서 아이콘**: 해당 이미지의 JSON 어노테이션 파일이 존재함
- 아이콘 없음: JSON 없음

---

### 클래스 관리

1. 우측 **Classes** 패널에서 `+ Add` 클릭
2. 클래스 이름 입력
3. 클릭으로 활성 클래스 선택 (볼드체로 표시)
4. 불필요한 클래스는 선택 후 `− Remove` (어노테이션이 없을 때만 삭제 가능)

---

### Draw 폴리곤 도구 (`D`)

- **좌클릭**: 꼭짓점 추가
- **첫 번째 꼭짓점 근처 클릭** 또는 **더블클릭**: 폴리곤 닫기
- **Enter**: 마스크 확정
- **Ctrl+Z**: 마지막 꼭짓점 취소
- **우클릭** 또는 **Esc**: 현재 폴리곤 취소

> 찍은 꼭짓점 좌표를 소수점 단위 그대로 저장합니다.

---

### Brush 도구 (`B`)

- **좌클릭 + 드래그**: 마스크 칠하기
- **우클릭 + 드래그**: 마스크 지우기
- `[` / `]`: 브러시 크기 조절
- **Enter**: 마스크 확정
- **Esc**: 취소

#### 레이블 편집 중 브러시 사용

Labels 패널에서 레이블을 선택한 뒤 `B`를 눌러 기존 마스크를 직접 수정할 수 있습니다.
편집 중에는 컨트롤포인트 점이 숨겨져 경계선이 잘 보입니다.

---

### AI 매직완드 도구 (`M`)

EdgeSAM 또는 SAM2 (Hiera-Tiny / Base+ / Large) 모델을 이용한 자동 마스크 생성입니다. 사이드바의 **AI Model** 드롭다운에서 모델을 전환할 수 있으며, 선택은 다음 실행에도 유지됩니다.

- **좌클릭**: 포함할 영역 지정 (초록 점)
- **우클릭**: 제외할 영역 지정 (빨간 점)
- `[` / `]`: 마스크 후보 선택 (1/3 ~ 3/3, 작은 것 → 큰 것)
- **Enter**: 마스크 확정
- **Esc**: 초기화

> 선택한 모델의 가중치(`download_weights.py --model <edgesam|sam2|sam2_base_plus|sam2_large>`)와 `onnxruntime` 패키지가 필요합니다.

---

### Pan (이동) 도구 (`H`)

- `H` 한 번: Pan 모드 진입
- `H` 다시 누름: 직전에 사용하던 도구로 복귀
- **Space** 누르는 동안: 임시 Pan 모드 (어떤 도구에서든 사용 가능)
- **마우스 휠**: 확대/축소

---

### 레이블 편집

Labels 패널에서 레이블 클릭 시 편집 모드 진입:

- **컨트롤포인트 드래그**: 꼭짓점을 마우스로 끌어 윤곽선 조정
- **`B` + 브러시 페인팅**: 마스크 직접 수정
- **Esc**: 편집 모드 종료
- **Delete 키** (Labels 목록 선택 중): 선택된 레이블 삭제
- 하단 **콤보박스**: 선택된 레이블의 클래스 변경

---

### 저장 형식 (LabelMe JSON)

- 이미지당 하나의 `.json` 파일 생성
- LabelMe Format
- 어노테이션이 없는 이미지는 JSON 파일을 생성하지 않음


---

### 감마 보정

- `G`: 감마 보정 ON/OFF
- View → Gamma Curve Setting: 커브 직접 조정

---

## 파일 구조

```
HyLabel/
├── run_hylabel.py          # 실행 진입점
├── download_weights.py     # AI 매직완드 모델(EdgeSAM/SAM2) 다운로드
├── requirements.txt        # 패키지 목록
└── labeler/
    ├── main.py             # 앱 초기화
    ├── window.py           # 메인 윈도우
    ├── canvas.py           # 캔버스 (그리기/편집)
    ├── mask_manager.py     # 마스크 저장/렌더링
    ├── coco_io.py          # JSON 저장/불러오기
    ├── models.py           # 데이터 모델
    ├── sam_worker.py       # EdgeSAM/SAM2 ONNX 추론
    ├── gamma_dialog.py     # 감마 커브 UI
    └── weights/            # AI 매직완드 모델 가중치 (다운로드 후 생성)
```
