from collections import defaultdict
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

# 색인화할 문서들
documents = [
    "한국의 수도는 서울입니다.",
    "한국의 인구는 약 5천만 명입니다.",
    "한국은 아시아 대륙에 위치해 있습니다.",
    "한국의 주요 산업은 전자제품 제조업입니다."
]

# 형태소 분석기 초기화
okt = Okt()

# 각 문서를 토큰화하고, 토큰의 등장 빈도를 계산하여 저장
tokenized_documents = []
token_freqs = defaultdict(lambda: defaultdict(int))

for doc in documents:
    tokens = okt.morphs(doc)  # 문서를 형태소 단위로 분리
    tokenized_documents.append(tokens)  # 토큰화된 문서를 저장
    for token in tokens:
        token_freqs[token][doc] += 1  # 토큰의 등장 빈도를 저장

# BM25 모델 학습을 위해 문서를 입력 형식으로 변환
corpus = []
for tokens in tokenized_documents:
    corpus.append([token for token in tokens])

# BM25 모델 초기화
bm25 = BM25Okapi(corpus)

# 학습된 BM25 모델 사용 예시
query = "한국인은 몇명인가요?"
tokenized_query = okt.morphs(query)
scores = bm25.get_scores(tokenized_query)

# 각 문서에 대한 검색 점수 출력
for doc_id, score in enumerate(scores):
    print(f"Document {doc_id + 1}: Score = {score}")
