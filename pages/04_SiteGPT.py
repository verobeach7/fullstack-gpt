from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# BeautifulSoup 임포트
from bs4 import BeautifulSoup
import streamlit as st
import requests
import pickle

llm = ChatOpenAI(
    temperature=0.1,
)

# template은 invoke가 실행되면 templatedmf format시켜줌
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    # st.write(answers)
    # List가 아닌 Dictionary가 반환되어야 함. 다음 체인(choose_answer function)의 입력 형식임.
    return {
        "question": question,
        # List 내 Dictionary로 저장
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    # 각각의 문서에는 answer, source, date가 리스트로 들어있기 때문에 이를 하나의 string으로 변환해줘야 함
    # get_answers의 반환 값 확인
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


# soup은 beautiful soup object임. 검색, 삭제 작업 등 수행 가능
# BeautifulSoup을 임포트 후 명시적으로 soup에 지정해주면 자동완성도 사용 가능
def parse_page(soup: BeautifulSoup):
    # document의 내용 커스터마이징
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        # .decompose는 하위 태그를 포함하여 모두 제거함
        header.decompose()
    if footer:
        footer.decompose()
    # 여기서 반환하는 값은 무엇이든지 page_content로써 document에 포함되게 됨
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    def is_serializable(obj):
        try:
            pickle.dumps(obj)  # 객체를 직렬화 시도
            return True  # 성공하면 직렬화 가능
        except (pickle.PickleError, TypeError):
            return False  # 실패하면 직렬화 불가능

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    # URL 커스터마이징
    loader = SitemapLoader(
        url,
        # parsing_function property를 이용하여 파싱을 위한 함수를 작동시킬 수 있음
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    # Embeddings를 캐싱하기 원하는 경우 sitemap에서 얻은 각각의 url마다 별도의 cache를 만들어야 함
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    # retriever는 invoke메서드를 가지고 있기 때문에 체인에서 바로 사용할 수 있음
    # retriever invoke가 실행되면 document들을 반환해줌
    retriever = vector_store.as_retriever()
    # if is_serializable(retriever):
    #     print("직렬화 가능")
    # else:
    #     print("직렬화 불가능")
    return retriever


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    # url에 .xml이 포함되어 있지 않은 경우 url을 재입력하게 함
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            # Streamlit에서 $는 텍스트를 변형시킴. 문자 그대로를 보여주기 위해서는 \ 붙여줘야 함.
            st.markdown(result.content.replace("$", "\$"))
