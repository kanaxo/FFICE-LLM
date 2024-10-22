import streamlit as st

def read_md_file(filename):
    with open(filename, 'r') as f:
        md_content = f.read()
    return md_content

st.set_page_config(
    layout="centered",
    page_title="Methodology"
)

st.title("Methodology")
md_string = read_md_file("method.md")
st.markdown(md_string)

