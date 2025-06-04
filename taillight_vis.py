import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ────────────────────────────────────────────────────────────────────
# 0) 경로 설정
BASE_DIR = "/mnt/d/Dataset/03_Bigdata/Tail_Light/round(20230417095611_Autocar_Taillight_100)_time(1681716180_1681716235)/meta"  # JSON과 이미지가 있는 폴더
IMG_DIR = "/mnt/d/Dataset/03_Bigdata/Tail_Light/round(20230417095611_Autocar_Taillight_100)_time(1681716180_1681716235)/sensor/camera(00)"
OUT_DIR = os.path.join(BASE_DIR, "annotated_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# 1) JSON 파일 로드

def load_json(fname):
    with open(os.path.join(BASE_DIR, fname), "r", encoding="utf-8") as f:
        return json.load(f)

dataset       = load_json("dataset.json")
frames        = load_json("frame.json")
frame_data    = load_json("frame_data.json")
annotations   = load_json("frame_annotation.json")
instances     = load_json("instance.json")
# (ego_pose, sensor, log 등은 2D 시각화에는 사용하지 않음)

# ────────────────────────────────────────────────────────────────────
# 2) 조회를 빠르게 하기 위한 인덱스(딕셔너리) 생성

# frame_data_uuid → frame_data 레코드 (단일 매핑)
frame_data_by_uuid = {fd["uuid"]: fd for fd in frame_data}

# instance_uuid → 인스턴스 메타정보 (category_name 등)
instance_info = {inst["uuid"]: inst for inst in instances}

# ────────────────────────────────────────────────────────────────────
# 3) 첫 10개 annotation을 순차적으로 처리

for idx, ann in enumerate(annotations[:550], start=1):
    # 3-1) frame_data_uuid로 이미지 파일 정보 얻기
    fd_uuid = ann["frame_data_uuid"]
    if fd_uuid not in frame_data_by_uuid:
        print(f"[!] frame_data_uuid '{fd_uuid}' 없음, 스킵")
        continue
    fd = frame_data_by_uuid[fd_uuid]

    # file_name + file_format → 실제 이미지 파일명 (예: "1681716180099674780.png")
    img_fname = f"{fd['file_name']}.{fd['file_format']}"
    img_path = os.path.join(IMG_DIR, img_fname)
    if not os.path.isfile(img_path):
        print(f"[!] 이미지 파일을 찾을 수 없습니다: {img_path}, 스킵")
        continue

    # 3-2) 이미지를 불러와서 Matplotlib 축 준비 (RGB)
    img = plt.imread(img_path)
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(img)
    ax.set_axis_off()

    # 3-3) 해당 annotation의 bbox와 attribute를 그리기
    bbox = ann["geometry"]["bbox_image2d"]  # [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = bbox
    width  = x_max - x_min
    height = y_max - y_min

    # (a) 바운딩박스 그리기
    rect = patches.Rectangle(
        (x_min, y_min),
        width,
        height,
        linewidth=2,
        edgecolor="lime",
        facecolor="none"
    )
    ax.add_patch(rect)

    # (b) 클래스명 (instance의 category_name) 표시
    inst_uuid = ann["instance_uuid"]
    cls_name = instance_info.get(inst_uuid, {}).get("category_name", "unknown")
    # 예: "dynamic_object.vehicle.car"
    short_cls = cls_name.split(".")[-1]  # 마지막 부분(ca r 등)
    ax.text(
        x_min,
        y_min - 4,
        short_cls,
        fontsize=10,
        color="white",
        bbox=dict(facecolor="lime", edgecolor="none", pad=1)
    )

    # (c) attribute 값들을 텍스트로 표시 (bbox 오른쪽 아래에 여러 줄로)
    attrs = ann.get("attribute", {})
    # 한 줄에 한 개씩 "키: 값" 형식으로 넣기
    attr_lines = [f"{k}: {v}" for k, v in attrs.items()]
    if attr_lines:
        # 텍스트 박스를 그릴 위치 (bbox 우측 아래 쪽으로 약간 띄워서)
        text_x = x_max + 2
        text_y = y_min + 2  # bbox 상단부에서 조금 아래로

        # 하나의 텍스트 덩어리로 합치기
        text_all = "\n".join(attr_lines)
        ax.text(
            text_x,
            text_y,
            text_all,
            fontsize=9,
            color="white",
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2)
        )

    # 3-4) 제목(어떤 annotation인지 식별용) 추가
    ax.set_title(f"Annot #{idx}  |  FrameData: {fd_uuid[:8]}..  |  Inst: {inst_uuid[:8]}..")

    # 3-5) 결과 이미지 저장
    out_fname = f"annot_{idx:02d}_{fd['file_name']}.png"
    out_path = os.path.join(OUT_DIR, out_fname)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"[+] 저장 완료: {out_path}")

print("=== 모든 작업 완료 ===")
