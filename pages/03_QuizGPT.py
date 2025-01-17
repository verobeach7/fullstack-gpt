from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


# docs를 하나의 string으로 변경
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# file을 작은 chunk로 쪼갠 후 docs 반환
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


# 사이드바
with st.sidebar:
    docs = None
    # selectbox를 이용해 file/wikipedia article 중 선택
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    # File 선택
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    # Wikipedia Article 선택
    else:
        # 사용자로부터 topic 받기
        topic = st.text_input("Search Wikipedia...")
        if topic:
            # WikipediaRetriver 설정
            # top_k_results: 몇 개의 문서를 가져올지 설정
            retriever = WikipediaRetriever(top_k_results=5)
            # 한국어 문서를 얻기를 원할 때 lang을 설정할 수 있음
            # retriever = WikipediaRetriever(
            #     top_k_results=5,
            #     lang="ko",
            # )
            with st.status("Searching Wikipedia..."):
                # topic 관련 문서 찾기
                docs = retriever.get_relevant_documents(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    # 프롬프트에 퀴즈 예시를 제공
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
            )
        ]
    )
    # invoke시 들어온 docs를 format_docs function을 이용해 string으로 변경
    chain = {"context": format_docs} | prompt | llm
    start = st.button("Generate Quiz")
    if start:
        chain.invoke(docs)
