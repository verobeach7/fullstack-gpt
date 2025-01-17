from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
import json


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
questions_chain = {"context": format_docs} | questions_prompt | llm

# json 형식으로 문제와 답을 formatting
formatting_prompt = ChatPromptTemplate.from_messages(
    # {{}}를 이용한 이유는 {}만 사용 시 데이터를 입력 받는 것으로 판단하기 때문
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
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
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

    start = st.button("Generate Quiz")
    if start:
        # questions_response = questions_chain.invoke(docs)
        # st.write(questions_response.content)
        # formatting_response = formatting_chain.invoke(
        #     {"context": questions_response.content}
        # )
        # st.write(formatting_response.content)

        # 위의 작업을 체인을 이용하여 간단하게 처리할 수 있음
        chain = {"context": questions_chain} | formatting_chain | output_parser

        response = chain.invoke(docs)
        st.write(response)
