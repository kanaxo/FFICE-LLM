### This handles all the back-end functions

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv
from helper_functions import llm # <--- This is the helper function that we have created 
import streamlit as st

from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv('.env')

# Pass the API Key to the OpenAI Client
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY']) #os.getenv('OPENAI_API_KEY')

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

vectordb = FAISS.load_local(
    "faiss_index_2", embeddings_model, allow_dangerous_deserialization=True
)

print("Database loaded.")

def create_qa_chain():
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model='gpt-4o-mini'),
        retriever=vectordb.as_retriever(
            search_kwargs={'k': 3}
            ),
        return_source_documents=True, # Make inspection of document possible
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

def process_user_prompt(query):
    qa_chain = create_qa_chain()

    print(f"Querying ChatGPT: {query}\n")
    ### TEST OUTPUT ###
    output = qa_chain.invoke(query)

    count_tokens_from_query(query, output)

    return output


def count_tokens_from_query(query, output):
    print("Get number of tokens: \n")
    ### OBTAIN NUMBER OF TOKENS USED ###
    # Combine the retrieved documents into a single context string
    context_string = "\n\n".join([doc.page_content for doc in output['source_documents']])  # Adjust according to your document structure
    # Get response only
    model_response = output['result']
    formatted_prompt = template.format(context=context_string, question=query)
    # Count tokens
    prompt_tokens = llm.count_tokens(formatted_prompt)
    output_tokens = llm.count_tokens(model_response)

    # Calculate total tokens
    total_tokens = prompt_tokens + output_tokens

    # Print the token counts
    print("Prompt Tokens:", prompt_tokens)
    print("Output Tokens:", output_tokens)
    print("Total Tokens:", total_tokens)

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

def process_subsequent_prompt(question, chat_history):
    print("\nchat_history: ", chat_history)
    refined_query = generate_refined_query(question, chat_history)
    print("\nOriginal Question:", question)
    print("\nrefined_query: ", refined_query)
    qa_chain = create_qa_chain()

    output = qa_chain.invoke(refined_query)
    count_tokens_from_query(question, output)
    return output