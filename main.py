from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote.messages import stream_response
from langchain_community.embeddings import OllamaEmbeddings

ollama_embeddings = OllamaEmbeddings(
    # model="nomic-embed-text",
    model="qwen2:7b-instruct-q8_0"
    # model="chatfire/bge-m3:q8_0" # BGE-M3
)

texts = [
    "김삼봉은 14살이며, 아주 촉망받는 무장이었다. 그는 조선시대 사람이고, 수학을 좋아했다. 키는 180cm, 몸무게는 85kg으로 칼과 창을 아주 잘 다루었다. 총 또한 좋아했다."
]
embedded_documents = ollama_embeddings.embed_documents(texts)

# 2. Chroma DB 초기화 및 임베딩 추가
persist_directory = "db"
vectorstore = Chroma(embedding_function=ollama_embeddings, persist_directory=persist_directory)

# 텍스트와 해당 임베딩을 함께 Chroma DB에 추가
vectorstore.add_texts(texts, embeddings=embedded_documents)

# 3. Chroma DB에 임베딩된 데이터를 저장 - 최신 버전에서는 persist() 호출 불필요
# vectorstore.persist()

# 4. Chroma DB에서 임베딩된 텍스트 검색
query = "김삼봉이 좋아했던 무기는?"
results = vectorstore.similarity_search(query, k=3)

# Ollama 모델을 불러옵니다.
llm = ChatOllama(model="qwen2:7b-instruct-q8_0")
# EEVE-Korean-10.8B:latest
# 프롬프트
prompt = ChatPromptTemplate.from_template("{topic} 에 대하여 간략히 설명해 줘.")

# 체인 생성
chain = prompt | llm | StrOutputParser()


for result in results:
    answer = chain.stream({"topic": result.page_content})
    print(answer)
    # 스트리밍 출력
    stream_response(answer)

# 간결성을 위해 응답은 터미널에 출력됩니다.
# answer = chain.stream({"topic": "김삼봉"})



#######################


# import logging
#
# from langchain.vectorstores import Chroma
# import requests
#
#
# # Ollama API 클래스 정의
# class OllamaAPI:
#     def __init__(self, base_url="http://localhost:11434"):  # 포트 번호를 11434로 설정
#         self.base_url = base_url
#
#     def embed(self, model_name, text):
#         url = f"{self.base_url}/api/embed"
#         payload = {
#             "model": model_name,
#             "input": text
#         }
#         response = requests.post(url, json=payload)
#
#         if response.status_code == 200:
#             return response.json()["embeddings"]
#         else:
#             raise Exception(f"Error: {response.status_code}, {response.text}")
#
#     def ask(self, model_name, question2, context=""):
#         url = f"{self.base_url}/api/generate"
#         payload = {
#             "model": model_name,
#             "prompt": question2,
#             "context": context,
#             "stream": False
#         }
#         response3 = requests.post(url, json=payload)
#
#         if response3.status_code == 200:
#             return response3.json()
#         else:
#             raise Exception(f"Error: {response3.status_code}, {response3.text}")
#
#
# # 1. Ollama를 사용한 임베딩 생성 및 저장
# def embed_paragraph_with_ollama(paragraph, persist_directory="db"):
#     ollama_api = OllamaAPI()
#     model_name = "qwen2:7b-instruct-q8_0"  # Ollama 모델 이름으로 설정
#
#     # Ollama를 사용해 문단 임베딩 생성
#     embedding = ollama_api.embed(model_name, paragraph)
#
#     # ChromaDB 초기화
#     vectorstore = Chroma(embedding_function=lambda x: embedding, persist_directory=persist_directory)
#
#     # 임베딩 추가
#     # 텍스트와 임베딩을 함께 추가
#     vectorstore.add_texts([paragraph], embeddings=[embedding])
#
#     # 임베딩 저장
#     vectorstore.persist()
#     return vectorstore
#
#
# # 2. 질문에 대한 유사 답변 검색
# def query_similar_responses_with_ollama(question, persist_directory="db"):
#     ollama_api = OllamaAPI()
#     model_name = "qwen2:7b-instruct-q8_0"
#
#     # ChromaDB 로드
#     vectorstore = Chroma(embedding_function=lambda x: ollama_api.embed(model_name, x),
#                          persist_directory=persist_directory)
#
#     # 유사 답변 검색
#     docs = vectorstore.similarity_search(question, k=3)
#
#     return [doc.page_content for doc in docs]
#
#
# # 3. Ollama를 이용하여 질문에 답변 생성
# def generate_responses_with_ollama(question, docs):
#     ollama_api = OllamaAPI()
#     model_name = "qwen2:7b-instruct-q8_0"  # 주어진 모델 이름으로 설정
#
#     responses = []
#     for doc in docs:
#         response = ollama_api.ask(model_name, question, context=doc)
#         responses.append(response['choices'][0]['text'])
#
#     return responses
#
#
# if __name__ == "__main__":
#     # 예시 문단 임베딩 및 저장 (Ollama 기반)
#     paragraph = "김삼봉은 14살이며, 아주 촉망받는 무장이었다. 그는 조선시대 사람이고, 수학을 좋아했다. 키는 180cm, 몸무게는 85kg으로 칼과 창을 아주 잘 다루었다. 총 또한 좋아했다."
#     vectorstore = embed_paragraph_with_ollama(paragraph)
#
#     # 예시 질문
#     question = "김삼봉이 좋아했던 무기들을 알려줘."
#
#     # 유사한 답변 검색 (Ollama 기반)
#     similar_responses = query_similar_responses_with_ollama(question)
#
#     # Ollama를 사용하여 답변 생성
#     ollama_responses = generate_responses_with_ollama(question, similar_responses)
#
#     # 결과 출력
#     for i, response in enumerate(ollama_responses):
#         print(f"Response {i + 1}: {response}")
