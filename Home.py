import streamlit as st
from langchain.prompts import PromptTemplate

# st.write는 AI로부터 받은 것이 무엇이든지 볼 수 있음

# write string
st.write("Hello")

# write array
st.write([1, 2, 3, 4])

# write dictionary
st.write({"x": 1})

# write class: Library에 있는 클래스에 대한 것을 볼 수 있음
# 클래스 내에 설명 문서가 작성되어 있는 경우 이도 볼 수 있음
# 클래스 내에 property를 확인할 수 있음
st.write(PromptTemplate)

p = PromptTemplate.from_template("xxxx")

st.write(p)

# Magic: 변수명만 적으면 자동으로 출력해줌
# 장점은 간단하게 변수명만 작성하면 되지만 단점은 이것이 화면에 출력한다는 의미를 코드에 담지 못함

a = [5, 6, 7]

d = {"y": 10}

a

d

PromptTemplate

p

option = st.selectbox("Choose your model", ("GPT-3", "GPT-4"))

st.write("You selected: ", option)
