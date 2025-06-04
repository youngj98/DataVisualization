import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) JSON 파일 경로와 이미지 경로 설정
json_path = "/home/young/LocalDataset/01_AdverseWeather/102.고정밀데이터_수집차량_악천후_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/Clip_000/Camera/Camera_Front/375_ND_000_CF_007.json"
# JSON 내부의 "imagePath" 필드를 따로 쓸 수도 있음
image_path = "/home/young/LocalDataset/01_AdverseWeather/102.고정밀데이터_수집차량_악천후_데이터/01-1.정식개방데이터/Validation/01.원천데이터/Clip_000/Camera/Camera_Front/375_ND_000_CF_007.jpg"
# 만약 같은 폴더에 이미지가 있다면 image_path = json_data["imagePath"]

# 2) 레이블별 색상 맵 정의 (BGR 형식)
label_colors = {
    "sky":          (128,  64, 128),
    "vegetation":   ( 60, 179,  75),
    "fence":        (  0,   0, 142),
    "static":       (220, 220, 220),
    "ground":       (152, 251, 152),
    "road":         (128,  64, 128),
    "guard rail":   (102, 102, 156),
    "traffic sign": (255, 255,   0),
    "pole":         (153, 153, 153),
    "car":          (  0,   0, 142),
    "bus":          (  0,   0,  92),
    "truck":        (  0,   0,  70),
    # 필요하다면 다른 레이블도 추가
}

# 3) JSON 로드
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# JSON 내부에서 imagePath, imageHeight, imageWidth 정보 얻기
image_path = json_data.get("imagePath", None)
img_h = json_data.get("imageHeight", None)
img_w = json_data.get("imageWidth", None)

# 4) 원본 이미지 로드 또는 빈 캔버스 생성
if image_path is not None:
    # 이미지가 존재할 경우
    img = cv2.imread(image_path)
    if img is None:
        # 이미지 파일을 못 찾으면 빈 검정 캔버스로 대체
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
else:
    # imagePath 필드가 비어있거나 없을 경우 빈 캔버스
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

# Overlay 용 마스크 생성 (투명도 있는 시각화를 위해)
overlay = img.copy()

# 5) 각각의 shape(폴리곤) 그리기
for shape in json_data["shapes"]:
    label = shape["label"]
    points = shape["points"]  # [[x1,y1], [x2,y2], ...]
    
    # label에 매칭되는 색상 가져오기 (없으면 흰색)
    color = label_colors.get(label, (255, 255, 255))
    
    # numpy 배열로 변환 (정수형)
    pts = np.array(points, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))  # OpenCV가 요구하는 형태로 reshape
    
    # 폴리곤 내부를 색으로 채우기
    cv2.fillPoly(overlay, [pts], color)

# 6) 투명도 조절 후 합성 (alpha: 투명도 비율, 0.0~1.0)
alpha = 0.5
cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# 7) Matplotlib을 이용해 BGR→RGB 변환 후 화면에 출력
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Segmentation Overlay")
plt.show()
