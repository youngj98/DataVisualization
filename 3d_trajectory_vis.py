import os
import glob
import json
import numpy as np
import open3d as o3d
import matplotlib
# ───────────────────────────────────────────────────────────────
# “Agg” 백엔드로 설정 → GUI 없이 이미지 파일로 저장
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 축

# ───────────────────────────────────────────────────────────────
# (1) JSON 파일들이 들어 있는 폴더 (사용자 환경에 맞게 수정)
JSON_DIR = "/mnt/d/Dataset/05_3D_DynamicObject_Trajectory_2024/scene_001/json"

# (2) 대응되는 PCD 파일들이 들어 있는 폴더 (사용자 환경에 맞게 수정)
PCD_DIR = "/mnt/d/Dataset/05_3D_DynamicObject_Trajectory_2024/scene_001/lidar/stitched"

# (3) 시각화 결과(PNG)를 저장할 폴더 (사용자 환경에 맞게 수정)
OUTPUT_DIR = "/mnt/d/Dataset/05_3D_DynamicObject_Trajectory_2024/sample2"

# (4) PCD 파일 확장자 (예: ".pcd" 혹은 ".bin")
PCD_EXTENSION = ".pcd"

# 반드시 존재하도록 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 3D 박스를 그릴 때 연결해야 할 8개 꼭짓점의 인덱스 페어
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face (roof)
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face (ground)
    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical pillars
]


# ───────────────────────────────────────────────────────────────
# 2) 하나의 JSON + PCD를 3D로 시각화하는 함수
def visualize_3d_boxes(json_path, pcd_dir, output_dir,
                       elev=30, azim=-60, zoom_scale=0.5, point_alpha=0.6):
    """
    - json_path: 하나의 라벨링 JSON 파일 경로
    - pcd_dir: JSON과 같은 이름으로 된 PCD 파일들이 모여 있는 폴더
    - output_dir: 결과 PNG를 저장할 폴더
    - elev, azim: 카메라 고도(elevation)와 방위(azimuth) 각도
    - zoom_scale: 전체 scene 범위 대비 몇 배만 보일지 결정 (0< zoom_scale <=1)
    - point_alpha: 포인트클라우드 점 투명도 (0~1)
    """
    basename = os.path.splitext(os.path.basename(json_path))[0]
    pcd_path = os.path.join(pcd_dir, basename + PCD_EXTENSION)
    if not os.path.isfile(pcd_path):
        print(f"[!] PCD 파일을 찾을 수 없습니다: {pcd_path}")
        return

    # --- JSON 로드 ------------------------------------------------
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    object_list = data.get("annotation_metadata", {}).get("object_list", [])
    if not object_list:
        print(f"[!] object_list가 비어 있습니다: {json_path}")
        return

    # --- PCD 로드 -------------------------------------------------
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)  # (N, 3) 배열

    # --- Matplotlib 3D 축 준비 ------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{basename}  |  Objects: {len(object_list)}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # --- 격자 끄기 -------------------------------------------------
    ax.grid(False)            # 메이저/마이너 모두 끄기
    # 아래 세 줄을 추가하면 축 배경(panels)에 흰색 배경/라인이 보이지 않게 됩니다.
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    ax._axis3don = False

    # --- 포인트클라우드 그리기 --------------------------------------
    if pts.shape[0] > 0:
        # 높이(z) 값을 컬러맵으로
        zs = pts[:, 2]
        sc = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=zs, cmap="viridis",
            s=0.5, alpha=point_alpha, linewidths=0
        )
        # cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label("Height (m)")

    # --- 3D 바운딩박스 그리기 ---------------------------------------
    all_box_verts = []  # (M,3) 배열로 쌓을 예정

    for obj in object_list:
        verts = obj.get("bbox_vertices", None)
        if verts is None or len(verts) != 8:
            continue
        corners = np.array(verts)  # (8,3)
        all_box_verts.append(corners)

        cls      = obj.get("class_name", "unknown")
        track_id = obj.get("track_id", "0")
        attrs    = obj.get("attribute", [])

        # 색상을 track_id 기반 해시로 뽑거나, class_name 길이에 따라 임의로 정할 수 있습니다.
        try:
            color_hash = (int(track_id) * 37) % 256
        except:
            color_hash = (len(cls) * 50) % 256
        cmap = plt.get_cmap("tab20")
        box_color = cmap(color_hash / 256)

        # 엣지 연결하여 3D 박스를 그림
        for i, j in BOX_EDGES:
            xi, yi, zi = corners[i]
            xj, yj, zj = corners[j]
            ax.plot(
                [xi, xj], [yi, yj], [zi, zj],
                c=box_color, linewidth=1.5
            )

        # 바운딩박스 중심에 레이블 표시
        center = obj.get("bbox_center", None)
        if center is not None:
            cx, cy, cz = center
            label = f"{cls.split('.')[-1]}#{track_id}"
            # if attrs:
            #     label += "\n" + ",".join(attrs)
            ax.text(
                cx, cy, cz + 3,
                label,
                fontsize=8,
                color="black",
                backgroundcolor="none",
                ha="center", va="bottom"
            )

    # --- 축 비율 균일화 --------------------------------------------
    if pts.shape[0] > 0:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        mid = (mins + maxs) / 2
        max_range = (maxs - mins).max() / 2
        
        zoomed_range = max_range * zoom_scale
        ax.set_xlim(mid[0] - zoomed_range, mid[0] + zoomed_range)
        ax.set_ylim(mid[1] - zoomed_range, mid[1] + zoomed_range)
        ax.set_zlim(mid[2] - zoomed_range, mid[2] + zoomed_range)
    # # (a) 포인트클라우드 범위
    # if pts.shape[0] > 0:
    #     xmin, ymin, zmin = pts.min(axis=0)
    #     xmax, ymax, zmax = pts.max(axis=0)
    # else:
    #     xmin = ymin = zmin = 0
    #     xmax = ymax = zmax = 0

    # (b) 바운딩박스 꼭짓점 범위 (있으면 포함)
    if all_box_verts:
        all_box_verts = np.vstack(all_box_verts)  # 각 (8,3) → 합쳐서 (8*M, 3)
        bxmin, bymin, bzmin = all_box_verts.min(axis=0)
        bxmax, bymax, bzmax = all_box_verts.max(axis=0)

        ax.set_xlim(bxmin, bxmax)
        ax.set_ylim(bymin, bymax)
        ax.set_zlim(bzmin, bzmax)

    # --- 시점 설정 ------------------------------------------------
    ax.view_init(elev=elev, azim=azim)

    # --- 저장 -----------------------------------------------------
    out_path = os.path.join(output_dir, basename + "_3d.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"저장됨: {out_path}")


# ───────────────────────────────────────────────────────────────
# 3) 전체 JSON 파일 일괄 처리

json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
print(f"[*] 총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
for idx, jf in enumerate(json_files, start=1):
    print(f"  ({idx}/{len(json_files)}) {os.path.basename(jf)}")
    visualize_3d_boxes(
        jf, PCD_DIR, OUTPUT_DIR,
        elev=90,    # 카메라 고도
        azim=-60,   # 카메라 방위
        zoom_scale=0.5,
        point_alpha=0.6
    )

print("=== 완료 ===")