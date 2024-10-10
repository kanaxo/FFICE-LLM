# Read and store the Masterplan in chroma_db and test out RAG
import os
from dotenv import load_dotenv
from helper_functions import llm # <--- This is the helper function that we have created 
import regex as re
import pymupdf4llm

from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv('.env')

# Pass the API Key to the OpenAI Client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# embedding model that we will use
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

# llm to be used in RAG pipeplines
# llm = ChatOpenAI(model='gpt-4o-mini')

# Get the MD text
md_text = pymupdf4llm.to_markdown("Doc 9965 Vol II Implementation Guidance.pdf")  # get markdown for all pages

# Split the text at '## 1 INTRODUCTION'
parts = md_text.split('## 1 INTRODUCTION')
cleaned_text = parts[1]

# Remove appendix A
# Regex pattern to remove everything between "APPENDIX A" and "APPENDIX B"
pattern = r"\nAPPENDIX A.*?(?=APPENDIX B)"

# Remove the content in APPENDIX A
cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)

# Remove appendix E
# Regex pattern to remove everything between "APPENDIX A" and "APPENDIX B"
pattern = r"\nAPPENDIX E.*?(?=APPENDIX F)"

# Remove the content in APPENDIX A
cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)

# Remove appendix H
# Regex pattern to remove everything between "APPENDIX A" and "APPENDIX B"
pattern = r"\nAPPENDIX H.*?(?=APPENDIX I)"

# Remove the content in APPENDIX A
cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)

# Regex pattern (matches 'II-' followed by one or more digits)
# footer_pattern = r"\n+II-\d+[\n\-]+[A-Z\s]+Manual on FF ICE"
header_pattern = r"\n[A-Z\s]+Manual on FF\s*\-*ICE\n"
footer_pattern = r"\nII-\d+\n"
break_pattern = r"\n\s*-----\n"

# Replace the matched pattern with the two uppercase letters together (no newlines)

# Remove footers
cleaned_text = re.sub(header_pattern, '', cleaned_text)
cleaned_text = re.sub(footer_pattern, '', cleaned_text)
cleaned_text = re.sub(break_pattern, ' ', cleaned_text)

with open("parsed_output.txt", "w",  encoding='utf-8', errors = 'ignore') as file:
    file.write(cleaned_text)

print("Cleaned text saved to file successfully.")

### TEXT SPLITTING ###
# That's it. It is this simple.
text_splitter = SemanticChunker(embeddings = embeddings_model)

# Spit Text
splitted_documents = text_splitter.create_documents([cleaned_text])

### CREATE DATABASE ###
# Create the vector database
vectordb = Chroma.from_documents(
    documents=splitted_documents,
    embedding=embeddings_model,
    collection_name="ffice", # one database can have multiple collections
    persist_directory="./vector_db"
)
print('saved to chroma database')
### BUILD CHAIN ####

# Build prompt
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
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    ChatOpenAI(model='gpt-4o-mini'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True, # Make inspection of document possible
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

### TEST OUTPUT ###
question = 'Is ATFM restriction part of the R/T element?'
output = qa_chain.invoke(question)
print(output)

### OBTAIN NUMBER OF TOKENS USED ###
# Step 1: Retrieve documents based on the question
retrieved_docs = vectordb.as_retriever(search_kwargs={'k': 3}).get_relevant_documents(question)
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

