from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
import torch
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
import pinecone

# CLIP 모델 및 프로세서 로딩
clip_processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")
clip_model = AutoModel.from_pretrained("koclip/koclip-base-pt")

# BM25 모델을 학습시킬 데이터
documents = [
    "고양이 사진", "강아지 사진", "고양이가 앉은 사진", "강아지가 앉은 사진"
]
# BM25 모델 초기화
okt = Okt()
bm25 = BM25Okapi([okt.morphs(doc) for doc in documents])

# 이미지 로딩
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# 텍스트 데이터 준비
texts = ["고양이 사진", "강아지 사진", "고양이가 앉은 사진", "강아지가 앉은 사진"]

# CLIP 이미지 특징 추출
inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
image_features = clip_model.get_image_features(inputs["pixel_values"])

# BM25 텍스트 특징 추출
query = "고양이 사진"  # 예시 쿼리
tokenized_query = okt.morphs(query)
scores = bm25.get_scores(tokenized_query)

# Pinecone에 접속
pinecone.init(api_key="YOUR_API_KEY")  # Pinecone API 키 입력

# Index 생성
index_name = "hybrid_search_index"
index = pinecone.Index(name=index_name, metric="cosine", shards=1)

# 이미지와 텍스트의 특성을 결합하여 Pinecone 인덱스에 저장
for idx, (image_feature, score) in enumerate(zip(image_features, scores)):
    vector = torch.cat([image_feature, torch.tensor(score).unsqueeze(0)])  # CLIP 이미지 특성과 BM25 점수를 결합
    index.upsert(ids=str(idx), vectors=vector.numpy())

# Pinecone 인덱스에 데이터가 추가되었음을 확인
print(f"Number of vectors in the Pinecone index: {index.describe_index()['size']}")
