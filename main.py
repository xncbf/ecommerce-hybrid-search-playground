from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
import torch

# 모델과 프로세서 로딩
processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")
model = AutoModel.from_pretrained("koclip/koclip-base-pt")


# 이미지 로딩
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# 텍스트 데이터 준비
texts = ["고양이 사진", "강아지 사진", "고양이가 앉은 사진", "강아지가 앉은 사진"]

# 이미지와 텍스트 처리
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 특징 추출
image_features = model.get_image_features(inputs["pixel_values"])  # 이미지 특징 추출
text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])  # 텍스트 특징 추출

# 유사성 계산
image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

# 유사성 계산
similarity = torch.matmul(text_features, image_features.T)  # 행렬곱을 통한 유사성 계산

# 유사성 점수 출력
print("Similarity scores:")
print(similarity)

