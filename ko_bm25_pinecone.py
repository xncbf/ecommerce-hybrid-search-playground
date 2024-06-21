from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
import numpy as np
from pinecone import Pinecone
import os
# 문서와 쿼리 설정
documents = [
    "한국의 수도는 서울입니다.",
    "한국의 인구는 약 5천만 명입니다.",
    "한국은 아시아 대륙에 위치해 있습니다.",
    "한국의 주요 산업은 전자제품 제조업입니다."
]


# 형태소 분석기 초기화
okt = Okt()

# 문서를 토큰화하고 BM25 모델 학습
tokenized_documents = [okt.morphs(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_documents)

# BM25 점수 벡터화
doc_vectors = [bm25.get_scores(okt.morphs(doc)) for doc in documents]

# BM25 벡터를 스파스 딕셔너리 형식으로 변환
def to_sparse_dict(scores):
    indices = list(range(len(scores)))
    values = scores.tolist()
    return {'indices': indices, 'values': values}

sparse_vectors = [to_sparse_dict(scores) for scores in doc_vectors]

# 임베딩 벡터 (예시로 임의의 벡터 사용)
dense_vectors = [np.random.rand(128).tolist() for _ in documents]  # 128차원 임베딩 벡터

# Pinecone API 초기화
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
from pinecone import ServerlessSpec

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

# 인덱스 생성
index_name = 'hybrid-index'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=128,
        metric='dotproduct',
        spec=spec
    )

index = pc.Index(index_name)

# 인덱스에 벡터 업로드
upserts = []
for i, (sparse, dense) in enumerate(zip(sparse_vectors, dense_vectors)):
    _id = str(i)
    meta = {"text": documents[i]}
    upserts.append({
        'id': _id,
        'sparse_values': sparse,
        'values': dense,
        'metadata': meta
    })

index.upsert(upserts)


# 쿼리 처리
query = "한국인은 몇명인가요?"
tokenized_query = okt.morphs(query)
query_sparse_vector = to_sparse_dict(bm25.get_scores(tokenized_query))
query_dense_vector = np.random.rand(128).tolist()  # 실제 환경에서는 적절한 임베딩을 사용

# Pinecone에서 쿼리 벡터로 검색
response = index.query(
    top_k=5,
    vector=query_dense_vector,
    sparse_vector=query_sparse_vector,
    include_metadata=True
)

# 검색 결과 출력
for match in response['matches']:
    print(f"문서 ID: {match['id']}, 점수: {match['score']}, 내용: {match['metadata']['text']}")

# 인덱스 삭제 (테스트 후)
# pinecone.delete_index(index_name)
