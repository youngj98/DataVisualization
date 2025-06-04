import os
import glob
import json
import numpy as np
import open3d as o3d
import matplotlib
# Headless 환경에서도 저장 가능하도록 Agg 백엔드 사용
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 축

# ───────────────────────────────────────────────────────────────────────────────
# 1) 경로 및 전역 변수 설정 (자신의 환경에 맞게 수정하세요)
JSON_DIR    = "/mnt/d/Dataset/05_3D_DynamicObject_Trajectory_2024/scene_001/json"
PCD_DIR     = "/mnt/d/Dataset/05_3D_DynamicObject_Trajectory_2024/scene_001/lidar/stitched"
OUTPUT_DIR  = "/mnt/d/Dataset/05_3D_DynamicObject_Trajectory_2024/sample2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PCD_EXTENSION = ".pcd"  # 또는 실제 PCD 확장자가 .bin 등이라면 ".bin"으로 수정

# 3D 바운딩박스를 그릴 때 사용할 엣지 인덱스 (bbox_vertices는 8개 꼭짓점)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
]

# 카메라 시점 설정 (원한다면 수정)
CAM_ELEV = 90   # 고도 각도
CAM_AZIM = -60  # 방위 각도

# zoom_scale: 글로벌 범위의 몇 배만큼 사용해서 실제 축 범위를 계산할지
# (0<zoom_scale<=1). 1.0이면 전 범위를 그대로 써서 그리기.
ZOOM_SCALE = 0.5

# 포인트클라우드 투명도 (0~1)
POINT_ALPHA = 0.6

# ───────────────────────────────────────────────────────────────────────────────
# 2) 모든 JSON+PCD를 순회하여 “글로벌(X/Y/Z) min/max”를 계산하는 함수

def compute_global_ranges(json_dir, pcd_dir, extension):
    """
    • json_dir: JSON 파일들이 모여 있는 폴더
    • pcd_dir: JSON 이름과 동일한 PCD 파일들이 모여 있는 폴더
    • extension: PCD 파일 확장자 (".pcd" 또는 ".bin" 등)

    반환: (xmin_all, xmax_all, ymin_all, ymax_all, zmin_all, zmax_all)
    """
    # 초기값 세팅: 매우 큰/작은 값으로
    xmin_all, ymin_all, zmin_all = np.inf, np.inf, np.inf
    xmax_all, ymax_all, zmax_all = -np.inf, -np.inf, -np.inf

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    print(f"[*] 총 {len(json_files)}개의 JSON 파일을 순회하며 전역 범위를 계산합니다.")

    for idx, json_path in enumerate(json_files, start=1):
        basename = os.path.splitext(os.path.basename(json_path))[0]
        pcd_path = os.path.join(pcd_dir, basename + extension)
        if not os.path.isfile(pcd_path):
            print(f"[!] PCD 파일이 없습니다: {pcd_path}. 스킵합니다.")
            continue

        # 1) 포인트클라우드 범위 갱신
        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)  # (N, 3)
        if pts.size > 0:
            xmin_all = min(xmin_all, float(np.min(pts[:, 0])))
            xmax_all = max(xmax_all, float(np.max(pts[:, 0])))
            ymin_all = min(ymin_all, float(np.min(pts[:, 1])))
            ymax_all = max(ymax_all, float(np.max(pts[:, 1])))
            zmin_all = min(zmin_all, float(np.min(pts[:, 2])))
            zmax_all = max(zmax_all, float(np.max(pts[:, 2])))

        # 2) 바운딩박스 범위 갱신
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        object_list = data.get("annotation_metadata", {}).get("object_list", [])
        for obj in object_list:
            verts = obj.get("bbox_vertices", None)
            if verts is None or len(verts) != 8:
                continue
            corners = np.array(verts)  # (8,3)
            xmin_all = min(xmin_all, float(np.min(corners[:, 0])))
            xmax_all = max(xmax_all, float(np.max(corners[:, 0])))
            ymin_all = min(ymin_all, float(np.min(corners[:, 1])))
            ymax_all = max(ymax_all, float(np.max(corners[:, 1])))
            zmin_all = min(zmin_all, float(np.min(corners[:, 2])))
            zmax_all = max(zmax_all, float(np.max(corners[:, 2])))

        if idx % 50 == 0 or idx == len(json_files):
            print(f"  ({idx}/{len(json_files)}) 처리 중... "
                  f"현재 범위 X[{xmin_all:.2f}, {xmax_all:.2f}] "
                  f"Y[{ymin_all:.2f}, {ymax_all:.2f}] Z[{zmin_all:.2f}, {zmax_all:.2f}]")

    # 마지막으로 반환
    return xmin_all, xmax_all, ymin_all, ymax_all, zmin_all, zmax_all


# ───────────────────────────────────────────────────────────────────────────────
# 3) “글로벌 범위”를 이용해 각 프레임을 동일 축으로 그리는 함수

def visualize_3d_boxes_fixed_axes(
        json_path, pcd_dir, output_dir,
        global_ranges,
        elev=30, azim=-60,
        zoom_scale=1.0, point_alpha=0.6
    ):
    """
    • json_path: 하나의 라벨링 JSON 파일 경로
    • pcd_dir: JSON과 같은 이름으로 된 PCD 폴더
    • output_dir: 결과 PNG 저장 폴더
    • global_ranges: compute_global_ranges() 반환값 튜플
        → (xmin_all, xmax_all, ymin_all, ymax_all, zmin_all, zmax_all)
    • elev, azim: 카메라 고도·방위 각도
    • zoom_scale: 글로벌 범위 대비 몇 배만 보일지 (0<zoom_scale<=1)
    • point_alpha: 포인트클라우드 투명도
    """
    basename = os.path.splitext(os.path.basename(json_path))[0]
    pcd_path = os.path.join(pcd_dir, basename + PCD_EXTENSION)
    if not os.path.isfile(pcd_path):
        print(f"[!] PCD 파일이 없습니다: {pcd_path}. 스킵합니다.")
        return

    xmin_all, xmax_all, ymin_all, ymax_all, zmin_all, zmax_all = global_ranges

    # --- JSON 로드 ------------------------------------------------------
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    object_list = data.get("annotation_metadata", {}).get("object_list", [])
    if not object_list:
        print(f"[!] object_list가 비어 있습니다: {json_path}")
        return

    # --- PCD 로드 -------------------------------------------------------
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)  # (N,3)

    # --- Matplotlib 3D 축 준비 -----------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{basename}  |  Objects: {len(object_list)}")
    ax.set_xlabel("")  
    ax.set_ylabel("")  
    ax.set_zlabel("")
    ax.set_xticks([]) 
    ax.set_yticks([])
    ax.set_zticks([])

    # 격자 및 패널 배경 제거
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax._axis3don = False

    # --- 포인트클라우드 그리기 ------------------------------------------
    if pts.shape[0] > 0:
        sc = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=pts[:, 2], cmap="viridis",
            s=0.5, alpha=point_alpha, linewidths=0
        )
        # 컬러바 생략

    # --- 3D 바운딩박스 그리기 --------------------------------------------
    for obj in object_list:
        verts = obj.get("bbox_vertices", None)
        if verts is None or len(verts) != 8:
            continue
        corners = np.array(verts)  # (8,3)

        cls      = obj.get("class_name", "unknown")
        track_id = obj.get("track_id", "0")
        attrs    = obj.get("attribute", [])

        # 박스 색상: track_id 기반 해시
        try:
            color_hash = (int(track_id) * 37) % 256
        except:
            color_hash = (len(cls) * 50) % 256
        cmap = plt.get_cmap("tab20")
        box_color = cmap(color_hash / 256)

        # 12개 엣지 연결
        for i, j in BOX_EDGES:
            xi, yi, zi = corners[i]
            xj, yj, zj = corners[j]
            ax.plot(
                [xi, xj], [yi, yj], [zi, zj],
                c=box_color, linewidth=1.5
            )

        # 바운딩박스 중심에 라벨 (원치 않으면 주석 처리)
        center = obj.get("bbox_center", None)
        if center is not None:
            cx, cy, cz = center
            label = f"{cls.split('.')[-1]}#{track_id}"
            ax.text(
                cx, cy, cz + (zmax_all - zmin_all)*0.02,
                label,
                fontsize=6,
                color="black",
                backgroundcolor="none",
                ha="center", va="bottom"
            )

    # --- 축 범위(zoom_scale 적용) --------------------------------------
    # 글로벌 범위를 mid ± (global_range * zoom_scale) 형태로 설정
    mid_x = (xmin_all + xmax_all) / 2.0
    mid_y = (ymin_all + ymax_all) / 2.0
    mid_z = (zmin_all + zmax_all) / 2.0

    half_x = (xmax_all - xmin_all) / 2.0
    half_y = (ymax_all - ymin_all) / 2.0
    half_z = (zmax_all - zmin_all) / 2.0

    # 글로벌 기준 가장 큰 half-range 계산
    max_half = max(half_x, half_y, half_z)

    # → zoomed_half = max_half * zoom_scale
    zoomed_half = max_half * zoom_scale

    ax.set_xlim(mid_x - zoomed_half, mid_x + zoomed_half)
    ax.set_ylim(mid_y - zoomed_half, mid_y + zoomed_half)
    ax.set_zlim(mid_z - zoomed_half, mid_z + zoomed_half)

    # --- 시점 설정 ------------------------------------------------------
    ax.view_init(elev=elev, azim=azim)

    # --- 파일 저장 ------------------------------------------------------
    out_path = os.path.join(output_dir, basename + "_3d.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"[V] 저장됨: {out_path}")


# ───────────────────────────────────────────────────────────────────────────────
# 4) 메인: 글로벌 범위 계산 후, 각 파일에 대해 동일 축으로 시각화

if __name__ == "__main__":
    # 1) 글로벌 min/max 범위 계산
    global_ranges = compute_global_ranges(JSON_DIR, PCD_DIR, PCD_EXTENSION)
    xmin_all, xmax_all, ymin_all, ymax_all, zmin_all, zmax_all = global_ranges
    print(f"\n==> 최종 글로벌 범위:")
    print(f"   X: [{xmin_all:.2f}, {xmax_all:.2f}]")
    print(f"   Y: [{ymin_all:.2f}, {ymax_all:.2f}]")
    print(f"   Z: [{zmin_all:.2f}, {zmax_all:.2f}]\n")

    # 2) 각 JSON 파일을 동일한 축 범위로 시각화
    json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    print(f"[*] 총 {len(json_files)}개의 JSON 파일을 동일 축으로 시각화합니다.")
    for idx, jf in enumerate(json_files, start=1):
        print(f"  ({idx}/{len(json_files)}) {os.path.basename(jf)}")
        visualize_3d_boxes_fixed_axes(
            jf, PCD_DIR, OUTPUT_DIR,
            global_ranges=global_ranges,
            elev=CAM_ELEV, azim=CAM_AZIM,
            zoom_scale=ZOOM_SCALE,
            point_alpha=POINT_ALPHA
        )

    print("=== 모든 프레임 시각화 완료 ===")
