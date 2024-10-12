### This handles all the front-end functions

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from logics.query_handler import process_user_prompt
from utility import check_password


# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="FFICE Query App"
)

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

# endregion <--------- Streamlit App Configuration --------->

st.title("FF-ICE Query App")

form = st.form(key="form")
form.subheader("Prompt")

user_prompt = form.text_area("Enter your prompt here", height=200)

if form.form_submit_button("Submit"):
    st.toast(f"User Input Submitted - {user_prompt}")
    output = process_user_prompt(user_prompt)

    st.write(output['result']) 
    with st.expander("Source documents"):
        documents = output['source_documents']
        for i in range(len(documents)):
            st.subheader(f"Document {i+1}")
            document = documents[i]
            st.write(document.page_content)
            st.divider()
    # st.write(output['source_documents'])
    