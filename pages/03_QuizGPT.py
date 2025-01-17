from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
import json
import os


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


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


# 프롬프트에 퀴즈 예시를 제공
questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant that is role playing as a teacher.
        
Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.

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
questions_chain = {"context": format_docs} | questions_prompt | llm

# json 형식으로 문제와 답을 formatting
formatting_prompt = ChatPromptTemplate.from_messages(
    # {{}}를 이용한 이유는 {}만 사용 시 데이터를 입력 받는 것으로 판단하기 때문
    # not a template variable임을 알리는 역할
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": False
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": False
                        }},
                        {{
                            "answer": "Green",
                            "correct": False
                        }},
                        {{
                            "answer": "Blue",
                            "correct": True
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": False
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": True
                        }},
                        {{
                            "answer": "Manila",
                            "correct": False
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": False
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": False
                        }},
                        {{
                            "answer": "2001",
                            "correct": False
                        }},
                        {{
                            "answer": "2009",
                            "correct": True
                        }},
                        {{
                            "answer": "1998",
                            "correct": False
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": True
                        }},
                        {{
                            "answer": "Painter",
                            "correct": False
                        }},
                        {{
                            "answer": "Actor",
                            "correct": False
                        }},
                        {{
                            "answer": "Model",
                            "correct": False
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""",
        )
    ]
)
formatting_chain = formatting_prompt | llm


# file을 작은 chunk로 쪼갠 후 docs 반환
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")

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


@st.cache_data(show_spinner="Making quiz...")
# _docs: _를 이용하여 데이터 서명을 만드는데 사용하지 않도록 함
# docs는 list로 변경 가능(Mutable)한 인자로 오류 발생.
# topic이라는 추가 인자를 주어 캐싱을 위한 서명으로 사용하도록 함
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


# 사이드바
with st.sidebar:
    docs = None
    topic = None
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
            # wiki_search를 이용하여 caching
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()
