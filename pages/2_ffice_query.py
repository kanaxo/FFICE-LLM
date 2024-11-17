import streamlit as st
from logics.query_handler import process_user_prompt, process_subsequent_prompt
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
st.write("Ask a query related to FF-ICE in the prompt below. If it doesn't give you what you want, you can try asking the chat again. Refresh to restart chat.")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_prompt := st.chat_input("Ask Me Anything!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # check if it's first query
        if len(st.session_state.messages) == 1:
            output = process_user_prompt(user_prompt)
        else:
            output = process_subsequent_prompt(user_prompt, st.session_state.messages)
        st.markdown(output['result'])
        with st.expander("Source documents"):
            documents = output['source_documents']
            for i in range(len(documents)):
                st.subheader(f"Document {i+1}")
                document = documents[i]
                st.write(document.page_content)
                st.divider()

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output['result']})

    # Button to clear the conversation history and reset the app
    # if st.button("Clear Conversation History"):
    #     st.session_state.messages = []
    #     st.toast("Conversation history cleared! Restarting the app...")
    #     st.rerun()
   
    