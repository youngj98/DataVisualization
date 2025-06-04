import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 축용

# 1) 파일 경로 설정
json_path = "/home/young/LocalDataset/01_AdverseWeather/102.고정밀데이터_수집차량_악천후_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/Clip_000/Lidar/Lidar_Roof/375_ND_000_LR_007.json"
pcd_bin_path = "/home/young/LocalDataset/01_AdverseWeather/102.고정밀데이터_수집차량_악천후_데이터/01-1.정식개방데이터/Validation/01.원천데이터/Clip_000/Lidar/Lidar_Roof/375_ND_000_LR_007.bin"  # 실제 .bin 파일 경로를 맞춰주세요

# 2) .bin 포인트클라우드 읽기 (KITTI 형식: float32 x,y,z,intensity)
points = np.fromfile(pcd_bin_path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]  # (N,3)

# 3) JSON 어노테이션 읽기
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]  # JSON 구조에 따라 키가 다를 수 있으니, 실제 필드를 확인하세요.

# 4) 바운딩박스 코너 계산 함수
def get_3d_bbox_corners(center, dims, yaw):
    """
    center: [x, y, z]
    dims:   [length, height, width]
    yaw:    라디안 단위 회전 (z축 기준)
    
    반환: (8, 3) 모양의 각 꼭짓점 좌표
    """
    l, h, w = dims
    x_c, y_c, z_c = center

    # ── 로컬 좌표계에서 미리 정의할 8개 코너 (yaw=0, 중심=(0,0,0))
    x_corners = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_corners = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    z_corners = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2])

    # 회전 행렬 (z축 기준 yaw)
    R = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [      0,             0,     1]
    ])

    corners = np.vstack((x_corners, y_corners, z_corners))   # (3,8)
    rotated = R @ corners                                    # (3,8)
    translated = rotated + np.array(center).reshape(3, 1)    # (3,8)
    return translated.T                                       # (8,3)

# 5) Matplotlib 3D로 그리기
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("PointCloud + 3D Bounding Boxes (Matplotlib)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 5-1) 포인트클라우드 산점도
# 너무 많은 점을 그리면 느리므로, 샘플링을 할 수도 있습니다. (예: xyz[::10])
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
           c='gray', s=0.5, alpha=0.5, label="PointCloud")

# 5-2) 각 어노테이션마다 바운딩박스 그리기
for ann in annotations:
    center = ann["3dbbox.location"]       # [x, y, z]
    dims   = ann["3dbbox.dimension"]      # [length, height, width]
    yaw    = ann["3dbbox.rotation_y"]     # 라디안
    category = ann.get("3dbbox.category", "unknown")

    # bbox 꼭짓점 (8개)를 구함
    corners = get_3d_bbox_corners(center, dims, yaw)  # (8,3)

    # 12개 엣지 연결 인덱스 (위쪽 4개 + 아래쪽 4개 + 수직 4개)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 상단 사각형
        (4, 5), (5, 6), (6, 7), (7, 4),  # 하단 사각형
        (0, 4), (1, 5), (2, 6), (3, 7)   # 수직 엣지
    ]

    # 카테고리별 색상
    color_map = {
        "car":   "r",
        "truck": "b",
        "bus":   "g"
    }
    c = color_map.get(category, "m")  # 정의되지 않은 카테고리는 보라색

    # 엣지마다 선을 그림
    for (i, j) in edges:
        x_line = [corners[i, 0], corners[j, 0]]
        y_line = [corners[i, 1], corners[j, 1]]
        z_line = [corners[i, 2], corners[j, 2]]
        ax.plot(x_line, y_line, z_line, c=c, linewidth=1.0)

ax.view_init(elev=90, azim=-90) # BEV 뷰
# 6) 축 비율 조정 (PointCloud가 왜곡 없이 보이도록)
xyz_range = np.ptp(xyz, axis=0)  # 각 축 범위 (ptp = max-min)
max_range = np.max(xyz_range)
mid_x = np.mean([np.min(xyz[:, 0]), np.max(xyz[:, 0])])
mid_y = np.mean([np.min(xyz[:, 1]), np.max(xyz[:, 1])])
mid_z = np.mean([np.min(xyz[:, 2]), np.max(xyz[:, 2])])

# 중심점 기준으로 반경을 max_range/2로 설정
ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
