import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ────────────────────────────────────────────────────────────────────
# 0) 경로 설정 부분만 실제 환경에 맞게 변경하세요.

# (1) JSON이 들어 있는 최상위 폴더 (라벨링 데이터)
ANNOTATION_ROOT = "/home/young/LocalDataset/134.차로_위반_영상_데이터/01.데이터/2.Validation/라벨링데이터/A"

# (2) JSON과 동일한 상대 경로에 대응되는 이미지를 찾을 최상위 폴더 (원천 데이터)
IMAGE_ROOT = "/home/young/LocalDataset/134.차로_위반_영상_데이터/01.데이터/2.Validation/원천데이터/A"

# (3) 결과를 저장할 최상위 폴더
OUTPUT_ROOT = "/home/young/LocalDataset/134.차로_위반_영상_데이터/01.데이터/2.Validation/annotated_output/A"

# JSON 하나당 최종 저장 파일명: 例) <폴더구조>/annotated_원본이름.png
# 필요시 확장자를 .jpg → .png 로 바꿔서 저장합니다.
# (4) 이미지 확장자 후보 (가장 흔한 순서대로 시도)
IMAGE_EXTENSIONS = [".jpg", ".png"]

# ────────────────────────────────────────────────────────────────────
# 1) 출력 폴더가 존재하지 않으면 전체 트리 생성
for root, dirs, _ in os.walk(ANNOTATION_ROOT):
    # 라벨링 데이터에서 하위 디렉터리 구조를 그대로 OUTPUT_ROOT로 복제
    rel = os.path.relpath(root, ANNOTATION_ROOT)  # e.g. "subfolder1/subsub1"
    out_folder = os.path.join(OUTPUT_ROOT, rel)
    os.makedirs(out_folder, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# 2) 인덱싱용 함수: 하나의 경로에서 JSON → 이미지 경로 매핑
def find_corresponding_image(annotation_json_path):
    """
    annotation_json_path 예시:
      /.../라벨링데이터/A/subfolder1/subsub1/[WHITE]72526A_174119_003.json

    반환값:
      "/.../원천데이터/A/subfolder1/subsub1/[WHITE]72526A_174119_003.jpg" 등
      이미지가 없다면 None 리턴
    """
    # (1) 절대 경로에서 ANNOTATION_ROOT까지의 상대 경로 (디렉터리 + 파일이름)
    rel_path = os.path.relpath(annotation_json_path, ANNOTATION_ROOT)
    # rel_path 예: "subfolder1/subsub1/[WHITE]72526A_174119_003.json"

    # (2) 확장자(.json) 떼어낸 base 이름과 하위 디렉터리 경로 분리
    rel_dir, rel_file = os.path.split(rel_path)
    base_name, _ = os.path.splitext(rel_file)  # → "[WHITE]72526A_174119_003"

    # (3) IMAGE_ROOT + rel_dir 내부에서 base_name + 확장자(.jpg/.png) 찾기
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(IMAGE_ROOT, rel_dir, base_name + ext)
        if os.path.isfile(candidate):
            return candidate

    return None

# ────────────────────────────────────────────────────────────────────
# 3) JSON 파일마다 처리하기
count = 0
for dirpath, _, filenames in os.walk(ANNOTATION_ROOT):
    for fname in filenames:
        if not fname.lower().endswith(".json"):
            continue

        count += 1
        json_path = os.path.join(dirpath, fname)

        # (1) JSON ↔ 이미지 경로 매핑
        img_path = find_corresponding_image(json_path)
        if img_path is None:
            print(f"[!] 이미지 파일을 찾을 수 없습니다: {json_path}")
            continue

        # (2) JSON 로드
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # JSON의 최상위 구조가 [ "dataID", "data_set_info" ] 형태라고 가정
        dataset_info = data.get("data_set_info", {})
        objects = dataset_info.get("data", [])  # annotation 객체들이 담긴 배열

        # (3) 이미지 불러오기 (Matplotlib 전용, RGB)
        img = plt.imread(img_path)

        # (4) Matplotlib Figure/Axis 준비
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_axis_off()

        # (5) 폴리곤 + 메타정보 그리기
        for obj in objects:
            # 5-1) 폴리곤 점들
            pts = obj["value"].get("points", [])
            if len(pts) < 3:
                continue
            poly_xy = [(p["x"], p["y"]) for p in pts]

            # 5-2) 메타정보 추출
            metainfo = obj["value"].get("metainfo", {})
            violation_type  = metainfo.get("violation_type", "")
            video_id        = metainfo.get("video_id", "")
            camera_channel  = metainfo.get("camera_channel", "")
            time_info       = metainfo.get("time_info", "")
            camera_number   = metainfo.get("camera_number", "")

            annotation_type = obj["value"].get("annotation", "")
            extra = obj["value"].get("extra", {})
            extra_value = extra.get("value", "")
            extra_label = extra.get("label", "")
            extra_color = extra.get("color", "#ff0000")

            obj_label = obj["value"].get("object_Label", {})

            # 폴리곤 그리기
            poly_patch = patches.Polygon(
                poly_xy,
                closed=True,
                linewidth=2,
                edgecolor=extra_color,
                facecolor="none",
                alpha=0.8
            )
            ax.add_patch(poly_patch)

            # 메타 텍스트: 폴리곤 첫 점 기준으로 약간 오프셋
            first_x, first_y = poly_xy[0]
            text_lines = [
                f"Violation: {violation_type}",
                f"VideoID: {video_id}",
                f"Camera: {camera_channel}-{camera_number}",
                f"Time: {time_info}",
                f"Type: {annotation_type}",
                f"{extra_label}: {extra_value}",
            ]
            # object_Label 내부 key:value 쌍도 추가
            for k, v in obj_label.items():
                text_lines.append(f"{k}: {v}")

            text = "\n".join(text_lines)
            ax.text(
                first_x + 3,
                first_y + 3,
                text,
                fontsize=9,
                color="white",
                va="top",
                ha="left",
                bbox=dict(facecolor=extra_color, edgecolor="none", alpha=0.7, pad=4)
            )

        # (6) 결과 저장: OUTPUT_ROOT + 동일한 상대 폴더 경로
        rel_dir = os.path.relpath(dirpath, ANNOTATION_ROOT)  # e.g. "subfolder1/subsub1"
        out_folder = os.path.join(OUTPUT_ROOT, rel_dir)
        os.makedirs(out_folder, exist_ok=True)

        base_name, _ = os.path.splitext(fname)
        out_png = os.path.join(out_folder, base_name + "_with_meta.png")
        plt.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"[{count:03d}] 저장 완료: {out_png}")

print("=== 전체 작업 완료 ===")
