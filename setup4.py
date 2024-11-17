# Read and store the Masterplan in chroma_db and test out RAG
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from helper_functions import llm
import regex as re
import pymupdf4llm
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.memory import ConversationBufferMemory
import faiss

def extract_text_from_pdf(pdf_path):
    # Get the MD text
    md_text = pymupdf4llm.to_markdown(pdf_path)  # get markdown for all pages
    return md_text

def preprocess_text(text):
    # Split the text at '## 1 INTRODUCTION'
    parts = text.split('## 1 INTRODUCTION')
    cleaned_text = parts[1]

    # Remove specific appendices
    cleaned_text = re.sub(r"\nAPPENDIX A.*?(?=APPENDIX B)", '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\nAPPENDIX E.*?(?=APPENDIX F)", '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\nAPPENDIX H.*?(?=APPENDIX I)", '', cleaned_text, flags=re.DOTALL)

    header_pattern = r"\n[A-Z\s]+Manual on FF\s*\-*ICE\n"
    footer_pattern = r"\nII-\d+\n"
    break_pattern = r"\n\s*-----\n"

    # Remove footers
    cleaned_text = re.sub(header_pattern, '', cleaned_text)
    cleaned_text = re.sub(footer_pattern, '', cleaned_text)
    cleaned_text = re.sub(break_pattern, ' ', cleaned_text)

    return cleaned_text

def split_documents(cleaned_text):

    # Create a recursive text splitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". "," ", ""]
    )

    # Split the cleaned text using the recursive splitter
    splitted_documents = recursive_splitter.create_documents([cleaned_text])

    print(f"Total number of chunks: {len(splitted_documents)}")

    list_of_tokencounts = []
    for doc in splitted_documents:
        # Count the tokens in the content of each page
        # then append the count to the list
        list_of_tokencounts.append(llm.count_tokens(doc.page_content))

    # Sum the token counts from all pages
    print(f"There are total of {np.sum(list_of_tokencounts)} tokens")
    print(f"The average number of tokens per chunk is {np.average(list_of_tokencounts)}")

    return splitted_documents

def create_faiss_db(splitted_documents, path, embeddings_model):
    # Create embeddings
    document_embeddings = embeddings_model.embed_documents([doc.page_content for doc in splitted_documents])
    # Initialize FAISS
    index = faiss.IndexFlatL2(len(document_embeddings[0]))  # Length of the embedding vectors
    vectordb = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    # Add documents and embeddings to FAISS
    vectordb.add_texts([doc.page_content for doc in splitted_documents])

    vectordb.save_local(path)

    return vectordb

def load_faiss_db(path, embeddings_model):
    return FAISS.load_local(
        path, embeddings_model, allow_dangerous_deserialization=True
    )

def build_chain(vectordb, template):
# Build prompt

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model='gpt-4o-mini'),
        retriever=vectordb.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True, # Make inspection of document possible
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain



def test_chain(qa_chain, question):
    ### TEST OUTPUT ###
    output = qa_chain.invoke(question)

    ### OBTAIN NUMBER OF TOKENS USED ###
    # Step 1: Retrieve documents based on the question
    # retrieved_docs = vectordb.as_retriever(search_kwargs={'k': 3}).get_relevant_documents(question)
    retrieved_docs = vectordb.similarity_search(query=question, k=3)
    # Step 2: Combine the retrieved documents into a single context string
    context_string = "\n\n".join([doc.page_content for doc in retrieved_docs])  # Adjust according to your document structure
    # Print the combined context string for inspection
    print("Combined Context String:\n", context_string)
    # Get response only
    model_response = output['result']
    formatted_prompt = template.format(context=context_string, question=question)
    # Step 5: Count tokens
    prompt_tokens = llm.count_tokens(formatted_prompt)
    output_tokens = llm.count_tokens(model_response)

    # Step 6: Calculate total tokens
    total_tokens = prompt_tokens + output_tokens

    # Print the token counts
    print("Prompt Tokens:", prompt_tokens)
    print("Output Tokens:", output_tokens)
    print("Total Tokens:", total_tokens)
    return output

load_dotenv('.env')

# Pass the API Key to the OpenAI Client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# embedding model that we will use
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

template = """You are an expert on the FF-ICE implementation guidance, which provides detailed instructions on implementing the latest air traffic management concept, including filing flight plans and operational procedures.

Your task is to:

1. Determine if the query is relevant to FF-ICE or air traffic management. If the query is not related, respond with:
'This query is not related to FF-ICE or air traffic management.'

2. If relevant, provide a helpful and direct answer based on the provided context. Don't say "This query is relevant to FF-ICE and air traffic management."

3. In addition to answering, suggest specific sections or tables from the FF-ICE guidance, including section numbers and titles, that support the answer.

4. If the specific answer cannot be found in the guidance, say:
'The specific answer is not available in the guidance.'
However, if you believe there are relevant sections or tables that can still provide useful context, list them with their section/table numbers and titles.

Important:
Always base your response on the provided context from the document. Avoid any speculative or unrelated information.
{context}
Question: {question}
Helpful Answer:"""

refinement_template = """
Use the chat history and the current question to create a more specific query for retrieving information.

Chat History:
{chat_history}

Original Question:
{question}

Refined Query:
"""
refinement_prompt = PromptTemplate(
    template=refinement_template,
    input_variables=["chat_history", "question"]
)

def generate_refined_query(question, chat_history):
    prompt = refinement_prompt.format(chat_history=chat_history, question=question)
    refined_query = llm.get_completion(prompt)  # Run the prompt through the LLM to get the refined query
    return refined_query.strip()

# Initialize memory to keep track of chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def test_chain_with_refinement(qa_chain, question):
    chat_history = memory.load_memory_variables({}).get("chat_history", "")
    print("\nchat_history: ", chat_history)
    refined_query = generate_refined_query(question, chat_history)
    print("\nOriginal Question:", question)
    print("\nrefined_query: ", refined_query)

    ### TEST OUTPUT ###
    output = qa_chain.invoke(refined_query)
    # save the refined query to the chat history
    memory.save_context({"input": refined_query}, {"output": output['result']})
    ### OBTAIN NUMBER OF TOKENS USED ###
    # Step 1: Retrieve documents based on the question
    # retrieved_docs = vectordb.as_retriever(search_kwargs={'k': 3}).get_relevant_documents(question)
    retrieved_docs = vectordb.similarity_search(query=refined_query, k=3)
    # Step 2: Combine the retrieved documents into a single context string
    context_string = "\n\n".join([doc.page_content for doc in retrieved_docs])  # Adjust according to your document structure
    # Print the combined context string for inspection
    print("Combined Context String:\n", context_string)
    # Get response only
    model_response = output['result']
    formatted_prompt = template.format(context=context_string, question=question)
    # Step 5: Count tokens
    prompt_tokens = llm.count_tokens(formatted_prompt)
    output_tokens = llm.count_tokens(model_response)

    # Step 6: Calculate total tokens
    total_tokens = prompt_tokens + output_tokens

    # Print the token counts
    print("Prompt Tokens:", prompt_tokens)
    print("Output Tokens:", output_tokens)
    print("Total Tokens:", total_tokens)
    return output

if __name__ == "__main__":

    if os.path.exists("faiss_index_2"):
        print("FAISS database already exists. Skipping preprocessing and creation.")
    else:
        # initialise the FAISS database
        print("FAISS database not found. Starting preprocessing and creation.")
        pdf_path = "Doc 9965 Vol II Implementation Guidance.pdf"
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned_text = preprocess_text(raw_text)

        with open("parsed_output.txt", "w", encoding='utf-8', errors='ignore') as file:
            file.write(cleaned_text)

        print("Cleaned text saved to file successfully.")

        splitted_documents = split_documents(cleaned_text)
        vectordb = create_faiss_db(splitted_documents, "faiss_index_2", embeddings_model)
        print("FAISS database created successfully.")

    # load the FAISS database
    vectordb = load_faiss_db("faiss_index_2", embeddings_model)
    print('saved to FAISS database')

    print("building chain")
    qa_chain = build_chain(vectordb, template)
    print("Testing chain")
    question = 'Is ATFM restriction part of the R/T element?'
    output = test_chain(qa_chain, question)
    print(output['result'])
    # save the question and answer to the chat history
    memory.save_context({"input": question}, {"output": output['result']})
    # test out the chain with refinement
    question = 'What is the purpose of attaching that to the R/T element?'
    output = test_chain_with_refinement(qa_chain, question)
    print(output['result'])

