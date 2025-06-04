import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# =============================================================================
# (0) 사용자 입력: PCD/JSON 파일 경로를 여기에서 지정
# =============================================================================
# 실제 경로(절대 혹은 상대)로 바꿔주세요.
pcd_path  = "/home/young/LocalDataset/173.자율주행_가상센서_시뮬레이션_데이터/01.데이터/Other/lidar/UR_SE_T1W1_U1_N01_RE01_126.pcd"
json_path = "/home/young/LocalDataset/173.자율주행_가상센서_시뮬레이션_데이터/01.데이터/Other/labeling/UR_SE_T1W1_U1_N01_RE01_126.json"

# =============================================================================
# (1) 3D 바운딩 박스 8개 코너 계산 함수
# =============================================================================
def get_3d_box_corners(center, size, yaw):
    """
    center: [x, y, z]
    size:   [length, width, height]  (l, w, h)
    yaw:    float (라디안, z축 회전 각도)
    return: np.ndarray of shape (8, 3)
    """
    cx, cy, cz = center
    l, w, h = size

    # 회전 전 로컬(local) 8개 코너 (center 기준)
    x_corners = np.array([ -l/2,  l/2,  l/2, -l/2,  -l/2,  l/2,  l/2, -l/2 ])
    y_corners = np.array([ -w/2, -w/2,  w/2,  w/2,  -w/2, -w/2,  w/2,  w/2 ])
    z_corners = np.array([ -h/2, -h/2, -h/2, -h/2,   h/2,  h/2,  h/2,  h/2 ])

    # z축 회전 행렬
    R = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0 ],
        [ np.sin(yaw),  np.cos(yaw), 0 ],
        [          0,            0, 1 ]
    ])

    corners = np.vstack((x_corners, y_corners, z_corners))  # shape=(3,8)
    corners_rot = R @ corners                               # (3,8)

    # 전역 좌표로 이동
    corners_rot[0, :] += cx
    corners_rot[1, :] += cy
    corners_rot[2, :] += cz

    return corners_rot.T  # (8,3)

# =============================================================================
# (2) JSON에서 3D 바운딩 박스 정보 읽기
# =============================================================================
def load_bboxes_from_json(json_path):
    """
    JSON 파일을 읽어서, 3D 정보가 있는 객체마다 8개 코너 좌표를 계산 후 리스트로 반환.
    반환 형식: [
        {
            "class": "vehicle",
            "corners": np.ndarray(shape=(8,3))
        },
        ...
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    bbox_list = []
    for obj in data.get("annotations", []):
        dim  = obj.get("dimension", [])
        loc  = obj.get("location", [])
        ori  = obj.get("orientation", [])

        # 3D 정보가 없다면 건너뜀
        if not (isinstance(dim, list) and len(dim) == 3 and
                isinstance(loc, list) and len(loc) == 3 and
                isinstance(ori, list) and len(ori) == 3):
            continue

        center = loc[:]      # [x, y, z]
        size   = dim[:]      # [length, width, height]
        yaw    = ori[2]      # orientation = [roll, pitch, yaw], 라디안

        corners = get_3d_box_corners(center, size, yaw)  # (8,3)
        bbox_list.append({
            "class": obj.get("class", "Unknown"),
            "corners": corners
        })

    return bbox_list

# =============================================================================
# (3) PCD 파일을 NumPy 배열로 로드
# =============================================================================
def load_pcd_as_numpy(pcd_path):
    """
    Open3D를 사용해 PCD를 읽고, (N,3) 형태의 NumPy 배열을 반환
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)  # (N, 3)
    return pts

# =============================================================================
# (4) Matplotlib 3D로 PCD + 3D 바운딩 박스 시각화
# =============================================================================
def visualize_pcd_and_boxes(pcd_path, json_path):
    # 1) PCD 로드
    points = load_pcd_as_numpy(pcd_path)
    print(f"[Info] Loaded PCD: {pcd_path} (총 점 수 = {points.shape[0]})")

    # 2) JSON에서 3D 바운딩 박스 정보 로드
    bboxes = load_bboxes_from_json(json_path)
    print(f"[Info] Loaded {len(bboxes)} 3D bounding boxes from JSON: {json_path}")

    # 3) Matplotlib 3D Figure 준비
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("PCD + 3D Bounding Boxes (Matplotlib)")

    # 4) 점군 산점도 그리기
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=0.5,     # 점 크기 (너무 크면 느려짐)
        c='gray',  # 점 색
        alpha=0.5, # 투명도
        linewidth=0
    )

    # 5) 각 바운딩 박스를 Line3DCollection으로 그리기
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # 하단 사각형
        (4,5), (5,6), (6,7), (7,4),  # 상단 사각형
        (0,4), (1,5), (2,6), (3,7)   # 수직 엣지
    ]

    for bbox in bboxes:
        corners = bbox["corners"]  # shape=(8,3)
        line_segments = []
        for (i, j) in edges:
            p1 = corners[i]
            p2 = corners[j]
            line_segments.append([p1, p2])

        lc = Line3DCollection(
            line_segments,
            colors='r',     # 바운딩 박스 선 색상 (빨강)
            linewidths=1.0,
            alpha=0.8
        )
        ax.add_collection3d(lc)

        # (선택) 클래스명 텍스트 표시: 박스 중심에 레이블 달기
        cx, cy, cz = np.mean(corners, axis=0)
        ax.text(
            cx, cy, cz + 0.1,      # z축 방향으로 살짝 띄워 텍스트 충돌 방지
            bbox["class"],
            color='red',
            fontsize=8
        )


    # 6) 축 레이블 및 비율 맞추기
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 왜곡 없이 보기 위해, 모든 축의 길이를 동일하게 설정
    x_min, x_max = points[:,0].min(), points[:,0].max()
    y_min, y_max = points[:,1].min(), points[:,1].max()
    z_min, z_max = points[:,2].min(), points[:,2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 7) 뷰 각도 조정 (필요 시 변경 가능)
    ax.view_init(elev=90, azim=-90) # BEV 뷰

    # 8) 플롯 화면에 표시
    plt.tight_layout()
    plt.show()

# =============================================================================
# (5) 스크립트 실행
# =============================================================================
if __name__ == "__main__":
    visualize_pcd_and_boxes(pcd_path, json_path)
