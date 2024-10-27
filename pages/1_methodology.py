import streamlit as st

st.set_page_config(
    layout="centered",
    page_title="Methodology"
)

st.title("Methodology")

def explain_rag_pipeline():
    st.subheader("RAG Pipeline Explanation")
    
    st.write("""
    The RAG (Retrieval-Augmented Generation) pipeline in our system consists of several key steps:

    1. **Document Loading**: The process begins with loading the PDF documents (namely the ICAO FF-ICE guidance document).

    2. **Text Splitting**: The loaded documents are then split into smaller, manageable chunks. This step is crucial for efficient processing and retrieval. We tried using semantic chunking and recursive splitting, but due to the complexity of the document, we decided to use a simple splitting method. Semantic chunking leads to very fragmented chunks (some very long, some very short). At least for recursive splitting, each chunk is controlled to be around 1000 words. We also put overlap to ensure that the context is not lost.

    3. **Embedding Creation**: Each text chunk is converted into a numerical representation (embedding) using a pre-trained language model. This allows for semantic understanding and comparison of text. We use the `text-embedding-3-small` model from OpenAI to embed the text chunks.

    4. **Vector Store**: The embeddings are stored in a vector database, which enables fast and efficient similarity searches. We use the `FAISS` library to store the embeddings as streamlit has some issues with `ChromaDB`.

    5. **User Query**: When a user inputs a question or query, it serves as the starting point for the retrieval process.

    6. **Query Embedding**: The user's query is also converted into an embedding using the same process as the document chunks.

    7. **Similarity Search**: The system searches the vector store to find the most relevant document chunks based on the similarity between the query embedding and the stored embeddings.

    8. **Context Retrieval**: The most relevant chunks are retrieved and combined to form the context for answering the query. We use the `k=3` for the number of chunks retrieved.

    9. **LLM Processing**: The retrieved context, along with the original query, is sent to a Large Language Model (LLM) for processing. We use the `gpt-4o-mini` model from OpenAI for this.

    10. **Response Generation**: The LLM generates a response based on the provided context and query, leveraging its pre-trained knowledge and the specific information from the retrieved documents.

    11. **Output**: Finally, the generated response is presented to the user as the answer to their query.

    This RAG pipeline allows for more accurate and context-aware responses by combining the power of pre-trained language models with specific, up-to-date information from the loaded documents.
             
    For more information about the setup, you can refer to `setup4.py`.
    """)

    st.image("pages/abc-llm.drawio.png", caption="RAG Pipeline Flowchart", use_column_width=True)

explain_rag_pipeline()
st.subheader("AI Crew Setup for Scenario Generation")
st.write("""
The AI Crew for scenario generation is set up using a structured approach with multiple agents and tasks. Here's an overview of the setup:

1. **Agents**:
   - **Synthesizer**: An agent responsible for expanding scenarios.
   - **FF-ICE Expert**: An agent with expertise in FF-ICE concepts and message types.
   - **Writer**: An agent that consolidates and summarizes information.

2. **Tasks**:
   - **Scenario Expansion**: The Synthesizer expands a given topic into a detailed FF-ICE scenario.
   - **Message Analysis**: The FF-ICE Expert analyzes the scenario and determines appropriate FF-ICE messages and services to use.
   - **Summarization**: The Writer consolidates the scenario and message exchanges into a comprehensive summary.

3. **Tools**:
   - **FF-ICE Tool**: A custom tool that provides information about FF-ICE message types and services.

4. **Workflow**:
   a. The Synthesizer expands the initial topic into a detailed scenario.
   b. The FF-ICE Expert analyzes the scenario and determines the appropriate FF-ICE messages and services.
   c. The Writer consolidates the information into a final summary.

5. **Execution**:
   - The Crew is initialized with the agents and tasks.
   - The process is kicked off with a given topic (user input).
   - Each task is executed in sequence, with the output of one task serving as input for the next.

This setup allows for a collaborative and specialized approach to scenario generation, leveraging the strengths of each agent to create comprehensive and accurate FF-ICE scenarios.

For more details on the implementation, you can refer to the `logics/scenario_builder.py` file.

""")

st.image("pages/abc-agents.drawio.png", caption="AI Crew Setup for Scenario Generation", use_column_width=True)