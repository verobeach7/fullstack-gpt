from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


# Streaming을 다루는 클래스 생성
class ChatCallbackHandler(BaseCallbackHandler):
    # 초기화 메소드: *arg는 argument(1,2,3,4), **kwarg는 keyword argument(a=1, b=2)
    def __init__(self, *args, **kwargs):
        self.message = ""

    # llm시작
    def on_llm_start(self, *args, **kwargs):
        # message_box 초기화
        self.message_box = st.empty()

    # llm종료
    def on_llm_end(self, *args, **kwargs):
        # message를 저장
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        # 기존 메시지에 계속 토큰을 붙여나감
        self.message += token
        # 추가되는 메시지를 메시지 박스에 마크다운으로 표시(업데이트)
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    # callbacks를 부를 때는 리스트로 제공
    callbacks=[ChatCallbackHandler()],
)


# Embedding Method
# 함수 위에 추가하는 것을 데코레이터라고 함
# st.cache_resource는 똑같은 파일이 들어오는 경우 embed_file method를 중복으로 처리하게 되어 효율성과 속도, 비용 모두 손해보기 때문에 재실행되는 것을 막아줌
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    # file 읽기
    file_content = file.read()
    # st.write(file_content)
    # file path 결정
    file_path = f"./.cache/files/{file.name}"
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")

    try:
        # st.write(file_path)
        # file_path에 w(rite)b(inary) 모드로 열기
        with open(file_path, "wb") as f:
            # file 내용을 기록
            f.write(file_content)
    except Exception as e:
        print(f"An error occurred: {e}")

    # cache directory 설정: 로컬에 파일 저장
    cache_basic_path = f"./.cache/embeddings/{file.name}"
    cache_directory = os.path.dirname(cache_basic_path)
    cache_path = LocalFileStore(f"./.cache/embeddings/{file.name}")

    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory, exist_ok=True)
        print(f"Cache Directory created: {cache_directory}")

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
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_path)
    # FAISS Vector Store를 이용하여 문서를 벡터로 저장
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    # Vectorstore에 저장된 값을 Retriever로 변환(리트리버로 변환해야 chain에서 사용 가능)
    retriever = vectorstore.as_retriever()
    # retriever에게 invoke하는 것은 문서에서 해당 내용을 검색하여 결과를 List of relavant docs로 반환
    # docs = retriever.invoke("ministry of truth")
    # st.write(docs) # 확인
    return retriever


# 다른 곳에서 또 사용되므로 별도의 함수로 구성
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# message를 화면에 보여주고, session_state에 저장하여 보관
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# session_state에 저장된 메시지를 화면에 보여줌, 저장은 비활성화
# 저장을 하는 경우 중복 저장됨
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


# 임베딩된 docs는 여러 개의 Document를 가지고 있으므로 내용만 뽑아서 2줄씩 띄어 줌으로써 AI가 더 잘 구분할 수 있게 함
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# 프롬프트 작성
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your file on the sidebar.
"""
)

# 파일 입력 창을 사이드바로 이동
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

# 파일을 불러온 경우 임베딩하고, 임베딩이 완료되면 준비완료 메시지를 보냄
# ai가 준비완료됐음을 나타내는 메시지는 저장할 필요 없음
if file:
    # embed_file 메소드를 통해 벡터화 된 문서들을 받음
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    # 화면에 저장된 메시지를 보여줌
    paint_history()
    # 사용자가 입력한 메시지를 보여주고 저장함
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        # 체인 생성
        chain = (
            {
                # RunnableLamda는 함수를 실행할 수 있게 해줌
                "context": retriever | RunnableLambda(format_docs),
                # 질문을 통과시켜 전달함
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        # chain.invoke할 때 ChatCallbackHandler가 작동함
        # AI로 하여금 박스를 생성하고 업데이트하게 하려면 chain.invoke를 st.chat_message 내부로 옮겨주기만 하면 됨
        # ai chat_message에서 invoke가 발생하기 때문에 ai 메시지 내부에 message_box를 생성하게 됨
        # 즉, chat_message 블록 내에서 체인을 호출
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    # 파일이 없거나 없어지는 경우 session_state를 초기화
    st.session_state["messages"] = []
