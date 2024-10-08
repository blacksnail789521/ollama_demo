import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path

model_local = ChatOllama(model="llama3.1:70b")
# model_local = ChatOllama(model="llama3.1")

# 1. Load Excel and Text Files

# Load the two Excel files into Pandas DataFrames
crm_df = pd.read_excel(Path("data", "CRM.xlsx"))
erp_df = pd.read_excel(Path("data", "ERP.xlsx"))

# Load the text file
with open(Path("data", "email.txt"), "r") as file:
    email_data = file.read()

# Convert Excel DataFrames to strings (or extract specific columns as needed)
crm_data = crm_df.to_string()
erp_data = erp_df.to_string()

# Combine all the document content into one list of strings
documents = [crm_data, erp_data, email_data]

# Split documents into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=7500, chunk_overlap=100
)
doc_splits = text_splitter.create_documents(documents)

# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
)
retriever = vectorstore.as_retriever()

# # 3. Test the pipeline before using RAG
# before_rag_template = """
# You need to process customer emails, evaluate customer information from CRM files, and generate quotations based on ERP files.
# Upon receiving an email, it first parses the email details, identifies the customer using the CRM data, and determines their tier,
# engagement history, and total spend. Next, it analyzes the provided CRM and ERP data structures to understand their fields and contents.
# You need to check for relevant columns like customer name, product ID, pricing, and stock levels.
# Once the data structures are understood, it can generate a custom quote, adjusting pricing and delivery times based on customer tier
# and available inventory. Finally, it drafts a personalized response email with the quotation and delivery details,
# providing a professional response based on the customer’s status and inquiry.
# """
# print("The response before using RAG:\n")
# before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
# before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
# print(before_rag_chain.invoke({"topic": "Process customer emails and generate quotations"}))

# 4. Test the pipeline after using RAG
print("\nThe response after using RAG:")

after_rag_template = """
You need to process customer emails, evaluate customer information from CRM files, and generate quotations based on ERP files. 
Upon receiving an email, it first parses the email details, identifies the customer using the CRM data, and determines their tier, 
engagement history, and total spend. Next, it analyzes the provided CRM and ERP data structures to understand their fields and contents. 
You need to check for relevant columns like customer name, product ID, pricing, and stock levels. 
Once the data structures are understood, it can generate a custom quote, adjusting pricing and delivery times based on customer tier 
and available inventory. Finally, it drafts a personalized response email with the quotation and delivery details, 
providing a professional response based on the customer’s status and inquiry.
Context: {context}
Question: {question}
"""

after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)

# print(after_rag_chain.invoke("How would you process customer emails and generate quotations based on CRM and ERP data?"))
print(
    after_rag_chain.invoke(
        "Please analyze the CRM and ERP structures and create a quote based on the inquiry."
    )
)
