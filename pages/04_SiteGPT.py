from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# BeautifulSoup 임포트
from bs4 import BeautifulSoup
import streamlit as st
import requests

# import requests

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


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    # URL 커스터마이징
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/blog\/).*",
        ],
        # parsing_function property를 이용하여 파싱을 위한 함수를 작동시킬 수 있음
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


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
        # 요청 응답이 xml인지 html인지 확인하는데 사용
        # response = requests.get(url)
        # print(response.text)

        # url로부터 페이지를 가져옴
        docs = load_website(url)
        st.write(docs)
