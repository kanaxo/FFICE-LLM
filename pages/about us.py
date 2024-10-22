import streamlit as st

def read_md_file(filename):
    with open(filename, 'r') as f:
        md_content = f.read()
    return md_content

st.set_page_config(
    layout="centered",
    page_title="About Us"
)

st.title("About Us")

st.write("""

         ## Project scope & objectives
        * Domain Area: Flight and Flow Information for a collaborative environment (FF-ICE) Implementation Guidance Help
        * Understanding ICAO Guidance for FF-ICE
        * What FF-ICE is:
            * FF-ICE is a next-generation flight planning mechanism to be rolled out globally by 2034. It will replace the current flight planning mechanism named FPL2012. 
        * This app aims to answer user queries about FF-ICE fast, and help them in visualising use cases that FF-ICE can enable.

         ## Use cases:        
        1. FF-ICE Chatbot
            Users can ask questions regarding FF-ICE implementation and the app shall extract and organize information from the implementation guidance to answer their queries. E.g. What are the 6 services in FF-ICE? This is implemented through a chat-based RAG. 
            Example question that was actually asked by one of our stakeholders: How can Air Traffic Flow Management restrictions be part of the R/T element?
        2. FF-ICE Scenario Builder
            Users can ask for a specific or a random scenario generation. These scenarios are useful in testing the services in a trial context. The app will build a scenario utilizing the 6 FF-ICE services.
          
         ## Data sources
        ICAO Doc 9965
          
         ## Features
        * shows source documents to allow reference to the document for more information
        * will tell user it doesn't know the answer, and to seek possible sections within the document for more information

""")

