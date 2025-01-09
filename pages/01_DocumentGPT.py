import time
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from pydantic import FilePath
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


# Embedding Method
def embed_file(file):
    # file 읽기
    file_content = file.read()
    # st.write(file_content)
    # file path 결정
    file_path = f"./.cache/files/{file.name}"
    # st.write(file_path)
    # file_path에 w(rite)b(inary) 모드로 열기
    with open(file_path, "wb") as f:
        # file 내용을 기록
        f.write(file_content)
    # cache directory 설정: 로컬에 파일 저장
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    # 파일 로더
    loader = UnstructuredFileLoader(file_path)
    # 쪼갠 문서들을 docs 리스트에 넣어둠. load_and_split은 List of docs 반환
    docs = loader.load_and_split(text_splitter=splitter)
    # OpenAIEmbeddings 메소드 사용
    embeddings = OpenAIEmbeddings()
    # 캐시에 임베딩하여 이미 캐시되어 있는 경우 작업을 생략. 없으면 임베딩.
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # FAISS Vector Store를 이용하여 문서를 벡터로 저장
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    # Vectorstore에 저장된 값을 Retriever로 변환
    retriever = vectorstore.as_retriever()
    # retriever에게 invoke하는 것은 문서에서 해당 내용을 검색하여 결과를 List of relavant docs로 반환
    # docs = retriever.invoke("ministry of truth")
    # st.write(docs) # 확인
    return retriever


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
"""
)

file = st.file_uploader(
    "Upload a .txt .pdf or .docx file",
    type=[
        "pdf",
        "txt",
        "docx",
    ],
)

if file:
    retriever = embed_file(file)
    s = retriever.invoke("describe victory mansion")
    s  # st.write를 생략하고 변수명만 적어도 출력함
