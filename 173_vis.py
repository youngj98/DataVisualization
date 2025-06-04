import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ────────────────────────────────────────────────────────────────────
# 1) 파일 경로 설정
json_path = "/home/young/LocalDataset/173.자율주행_가상센서_시뮬레이션_데이터/01.데이터/Other/labeling/UR_SE_T1W1_U1_N01_RE01_125.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

img_path = "/home/young/LocalDataset/173.자율주행_가상센서_시뮬레이션_데이터/01.데이터/Other/image/UR_SE_T1W1_U1_N01_RE01_125.png"

# ────────────────────────────────────────────────────────────────────
# 2) 원본 이미지 로드 (Matplotlib용으로 RGB로 읽음)
#    OpenCV 대신 plt.imread()를 써서 바로 NumPy 배열(RGB)로 만듭니다.
img = plt.imread(img_path)
h, w, _ = img.shape

# ────────────────────────────────────────────────────────────────────
# 3) JSON에서 바운딩박스 정보 읽어오기
annotations = data["annotations"]
# 각 ann에는 "bbox": [x_min, y_min, x_max, y_max], "class": 문자열 형태로 있다고 가정

# ────────────────────────────────────────────────────────────────────
# 4) Matplotlib으로 시각화
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)
ax.set_axis_off()

# 클래스별 색상을 지정하고 싶으면 아래에 맵을 정의해 주세요.
color_map = {
    "vehicle":   "lime",   # 예: 노랑-녹색 계통
    "policeCar": "cyan",
    "ambulance": "magenta",
    # 없으면 기본 빨간색으로 사용
}

for ann in annotations:
    bbox = ann["bbox"]  # [x_min, y_min, x_max, y_max]
    cls  = ann["class"]

    x_min, y_min, x_max, y_max = bbox
    width  = x_max - x_min
    height = y_max - y_min

    # (1) Rectangle 그리기
    edge_color = color_map.get(cls, "red")
    rect = patches.Rectangle(
        (x_min, y_min),
        width,
        height,
        linewidth=2,
        edgecolor=edge_color,
        facecolor="none"
    )
    ax.add_patch(rect)

    # (2) 클래스명 텍스트 표시 (상자 위쪽에)
    ax.text(
        x_min,
        y_min - 4,
        cls,
        fontsize=12,
        color="white",
        bbox=dict(facecolor=edge_color, edgecolor="none", pad=1)
    )

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────
# 5) 파일로 저장하고 싶으면 아래 주석을 해제하세요.
# out_path = "UR_SE_T1W1_U1_N01_RE01_001_with_bbox_matplotlib.png"
# fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
# print(f"바운딩박스가 그려진 이미지를 저장했습니다: {out_path}")