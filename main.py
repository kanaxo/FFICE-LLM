### This handles all the front-end functions
import streamlit as st

def read_md_file(filename):
    with open(filename, 'r') as f:
        md_content = f.read()
    return md_content

st.set_page_config(
    layout="centered",
    page_title="About Us"
)

st.title("About App")

st.expander("""

IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.

""")

st.write("""

         ## Project scope & objectives
        * Objective: Allow users to understand more about International Civil Aviation Organisation (ICAO) flight planning concept called Flight and Flow Information for a collaborative environment (FF-ICE). 
        * What FF-ICE is:
            * FF-ICE is a next-generation flight planning mechanism to be rolled out globally by 2034. It will replace the current flight planning mechanism named FPL2012. 
            * FF-ICE is meant to overcome the limitations of the current flight planning mechanism, such as the limited data fields in the current flight plan, and the lack of feedback mechanism after the flight plan is submitted, limiting negotiations between air navigation service providers (ANSPs) like Civil Aviation Authority of Singapore (CAAS) and airlines like Singapore Airlines (SIA).
        * This app aims to answer user queries about FF-ICE fast, and help them in visualising use cases that FF-ICE can enable.
        * For more information about FF-ICE, you can visit the [official ICAO FF-ICE page](https://www.icao.int/airnavigation/ffice/Pages/default.aspx).

         ## Use cases:        
        1. **FF-ICE Chatbot**: 
         
            * Users can ask questions regarding FF-ICE implementation and the app shall extract and organize information from the implementation guidance to answer their queries. E.g. What are the 6 services in FF-ICE? This is implemented through a chat-based RAG. 
            * Example question that was actually asked by one of our stakeholders: `How can Air Traffic Flow Management restrictions be part of the R/T element?`
        2. **FF-ICE Scenario Builder**:
         
            * Users can ask for a specific or a random scenario generation. The app will build a scenario utilizing the 6 FF-ICE services.
            * We need to build scenarios that are useful to showcase the capabilities of FF-ICE, especially when we do operational trials with industry stakeholders to convince them to adopt FF-ICE.
          
         ## Data sources
        * **ICAO Doc 9965** releasing in 2024, is the main guidance document for FF-ICE.
          
         ## Features
         ### FF-ICE Chatbot
        * Chatbot shows source documents to allow reference to the document for more information
        * Chatbot will tell user it doesn't know the answer, and to seek possible sections within the document for more information
         
        ### FF-ICE Scenario Builder
        * Scenario Builder builds a story through the use of AI crew to illustrate a future use case of FF-ICE, and features a table that shows some sample messages that can be sent using the 6 FF-ICE services.
         
        ### Example of a chat query and response:
        
""")

st.image("pages/ffice_chat_example.png", caption="FF-ICE Chat Example", use_column_width=True)

st.write("""
    ### Example of a scenario generation:
""")

st.image("pages/ffice_scenario_example.png", caption="FF-ICE Scenario Example", use_column_width=True)
st.subheader("FF-ICE Scenario Disclaimer")
st.write("""
As seen in the example, the scenario builder is very inaccurate. A lot of messages do not exist in the ICAO guidance document, or the sequence of messages are just not correct. Due to the complexity of the scenario generation, such as needing to deal with logical processes (e.g. a filing status is only available when the flight plan is filed), AI crew is not able to generate a perfect scenario yet.
""")

