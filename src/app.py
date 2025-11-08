import streamlit as st
from rag_core import get_rag_chain

st.set_page_config(page_title="AutoTechQA", page_icon=":Truck:", layout="wide")
st.title("AutoTechQA : Trucks Manual Question Answering")

@st.cache_resource
def load_rag_chain():
    rag_chain = get_rag_chain()
    return rag_chain

user_question = st.text_input("Enter your question about trucks:")
if user_question:
    with st.spinner("Searching manuals for answer..."):
        rag_chain = load_rag_chain()
    response = rag_chain.invoke(user_question)
    st.subheader("Answer:")
    st.write(response)