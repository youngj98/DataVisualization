import os
import glob
import cv2

def images_to_video(
    image_dir: str,
    image_ext: str,
    output_path: str,
    fps: int = 10
):
    """
    • image_dir: 이미지들이 들어 있는 폴더 경로
    • image_ext: 이미지 확장자 (예: "png" 또는 "jpg")
    • output_path: 생성할 비디오 파일 경로 (예: "output.mp4")
    • fps: 초당 프레임 수
    """

    # 1) 이미지 파일 목록 읽기 (정렬)
    pattern = os.path.join(image_dir, f"*.{image_ext}")
    image_paths = sorted(glob.glob(pattern))
    if len(image_paths) == 0:
        print(f"[!] '{image_dir}' 폴더에 '*.{image_ext}' 파일이 없습니다.")
        return

    # 2) 첫 번째 이미지를 읽어서 영상 크기 결정
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        print(f"[!] 첫 번째 이미지를 읽을 수 없습니다: {image_paths[0]}")
        return
    height, width, channels = first_frame.shape
    print(f"비디오 해상도: {width}x{height}, 채널: {channels}")

    # 3) VideoWriter 생성 (코덱: mp4v → .mp4 파일)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4) 이미지들을 순서대로 비디오에 프레임으로 추가
    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[경고] 프레임을 읽을 수 없습니다: {img_path} (스킵)")
            continue

        # 이미지 크기가 첫 프레임과 다르면 리사이즈
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        video_writer.write(frame)

        if idx % 50 == 0:
            print(f"  ▶ 프레임 작성 중: {idx+1}/{len(image_paths)}")

    # 5) 마무리
    video_writer.release()
    print(f"[완료] '{output_path}' 파일이 생성되었습니다.")


if __name__ == "__main__":
    # ===============================
    # 사용자가 수정해야 할 부분
    # ===============================
    IMAGE_FOLDER = "/mnt/d/Dataset/05_3D_DynamicObject_Trajectory_2024/sample2"  # 이미지들이 저장된 폴더
    IMAGE_EXT    = "png"                                # 이미지 확장자 (png, jpg 등)
    OUTPUT_VIDEO = "/mnt/d/Dataset/05_3D_DynamicObject_Trajectory_2024/sample2/output.mp4"  # 생성할 비디오 파일 경로
    FPS = 10  # 초당 프레임 수 (원하는 값으로 변경 가능)
    # ===============================

    images_to_video(IMAGE_FOLDER, IMAGE_EXT, OUTPUT_VIDEO, fps=FPS)
