from langchain.document_loaders import SitemapLoader
import streamlit as st

# import requests

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    # SitemapLoader는 sitemap.xml에 들어있는 모든 페이지로부터 문서를 가져옴
    loader = SitemapLoader(url)
    # 초당 요청 수를 제한하여 차단 정책이나 속도 제한 정책 위반을 방지
    loader.requests_per_second = 5
    docs = loader.load()
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

        # url로부터 모든 페이지를 가져옴
        docs = load_website(url)
        st.write(docs)
