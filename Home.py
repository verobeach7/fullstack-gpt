import streamlit as st

# 브라우저의 상단 탭의 이모지와 이름 설정 가능
st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🍟",
)

st.title("FullstackGPT Home")

# 페이지를 만들기 위해 반드시 'pages' 폴더를 만들어야 함
# 그 안에 파일을 만들면 자동으로 streamlit이 사이드바에 page를 구성해줌
# 사이드바에 순서를 변경하고 싶은 경우 pages 내의 파일의 앞에 '01_' 등 숫차로 정렬해주면 됨

st.markdown(
    """
# Hello!
            
Welcome to my FullstackGPT Portfolio!
            
Here are the apps I made:
            
- [x] [DocumentGPT](/DocumentGPT)
- [x] [PrivateGPT](/PrivateGPT)
- [x] [QuizGPT](/QuizGPT)
- [x] [SiteGPT](/SiteGPT)
- [x] [MeetingGPT](/MeetingGPT)
- [ ] [InvestorGPT](/InvestorGPT)
"""
)
