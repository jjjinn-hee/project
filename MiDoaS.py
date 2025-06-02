import torch
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# MiDaS 모델 로드
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # or "MiDaS_small"
midas.eval()
 
# 모델에 맞는 transform 준비
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
 
# 입력 이미지 로드
img = cv2.imread("KakaoTalk_20250530_233351241_28")  # 빗물받이 사진
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 전처리 및 추론
input_batch = transform(img)
 
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
 
depth_map = prediction.cpu().numpy()
 
# 결과 시각화
plt.imshow(depth_map, cmap='inferno')
plt.colorbar(label='Relative Depth')
plt.title("Depth Map by MiDaS")
plt.axis('off')
plt.show()
