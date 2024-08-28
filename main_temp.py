from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
import requests


# Ollama API 클래스 정의
class OllamaAPI:
    def __init__(self, base_url="http://localhost:11434"):  # 포트 번호를 11434로 설정
        self.base_url = base_url

    def ask(self, model_name, question2, context=""):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": question2,
            "stream": False
        }
        response3 = requests.post(url, json=payload)

        if response3.status_code == 200:
            return response3.json()
        else:
            raise Exception(f"Error: {response3.status_code}, {response3.text}")


# 1. SBERT 임베딩 생성 및 저장
def embed_paragraph(paragraph2, persist_directory="db"):
    # SBERT 모델 로드
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    # 문단 임베딩
    embeddings = model.encode([paragraph2])

    # ChromaDB 초기화
    vectorstore1 = Chroma(embedding_function=lambda x: embeddings, persist_directory=persist_directory)

    # 임베딩 추가
    vectorstore1.add_texts([paragraph2])

    # 임베딩 저장
    vectorstore1.persist()
    return vectorstore1


# 2. 질문에 대한 유사 답변 검색
def query_similar_responses(question4, persist_directory="db"):
    # SBERT 모델 로드
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    # ChromaDB 로드
    vectorstore2 = Chroma(embedding_function=lambda x: model.encode(x), persist_directory=persist_directory)

    # 유사 답변 검색
    docs = vectorstore2.similarity_search(question4, k=3)

    return [doc.page_content for doc in docs]


# 3. Ollama를 이용하여 질문에 답변 생성
def generate_responses_with_ollama(question5, docs):
    ollama_api = OllamaAPI()
    model_name = "qwen2:7b-instruct-q8_0"  # 주어진 모델 이름으로 설정

    responses = []
    for doc in docs:
        response1 = ollama_api.ask(model_name, question5, context=doc)
        responses.append(response1['choices'][0]['text'])

    return responses


if __name__ == "__main__":
    # 예시 문단 임베딩 및 저장
    paragraph = "김삼봉은 14살이며, 아주 촉망받는 무장이었다. 그는 조선시대 사람이고, 수학을 좋아했다. 키는 180cm, 몸무게는 85kg으로 칼과 창을 아주 잘 다루었다. 총 또한 좋아했다.."
    vectorstore = embed_paragraph(paragraph)

    # 예시 질문
    question = "김삼봉이 좋아했던 무기들을 알려줘."

    # 유사한 답변 검색
    similar_responses = query_similar_responses(question)

    # Ollama를 사용하여 답변 생성
    ollama_responses = generate_responses_with_ollama(question, similar_responses)

    # 결과 출력
    for i, response in enumerate(ollama_responses):
        print(f"Response {i + 1}: {response}")
