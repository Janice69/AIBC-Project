import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# --------------------------
# 1. Load OpenAI API Key from Text File
# --------------------------
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access your API key
api_key_path = r"C:\streamlit_projects\myenv\pages\OpenAI_Key.txt"  # Path to your OpenAI key file
openai_api_key = os.getenv("OPENAI_API_KEY")

# Use the key where needed
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)


# Read the OpenAI API key from the file
try:
    with open(api_key_path, 'r', encoding='utf-8') as file:
        openai_api_key = file.read().strip()  # Read and strip any extra whitespace
except FileNotFoundError:
    st.error("OpenAI API key file not found. Please check the file path.")
    st.stop()  # Stop execution if the file is not found

# --------------------------
# 2. Load and Process Text File
# --------------------------
# Path to your text file
txt_file_path = r"C:\streamlit_projects\myenv\HDB_Resale_Procedures.txt"

# Load the text from the file
with open(txt_file_path, 'r', encoding='utf-8') as file:
    hdb_resale_text = file.read()

# --------------------------
# 3. Split Text into Chunks
# --------------------------
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)

documents = text_splitter.split_text(hdb_resale_text)

# Convert into list of dictionaries as required for Chroma
docs_for_embedding = [{"text": doc} for doc in documents]

# --------------------------
# 4. Create Embeddings and Vector Store
# --------------------------
# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Create a Chroma vector store from the documents
vectorstore = Chroma.from_texts(
    [doc['text'] for doc in docs_for_embedding],
    embeddings,
    collection_name='hdb_resale_procedure'
)

st.success("Vector store created successfully.")

# --------------------------
# 5. Set Up Retrieval QA Chain
# --------------------------
# Define a custom prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.

{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Initialize the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Initialize the LLM
llm = OpenAI(
    temperature=0, 
    openai_api_key=openai_api_key
)

# Create the Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 'stuff' is a simple chain type; adjust as needed
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

st.success("Retrieval QA chain is set up.")

# --------------------------
# 6. Streamlit Chatbot Interface
# --------------------------
st.set_page_config(page_title="HDB Resale Procedure Chatbot")
st.title("HDB Resale Procedure Chatbot")
st.write("Ask any question related to the HDB Resale Procedure, and I'll provide you with an answer!")

user_question = st.text_input("Your Question:")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Fetching answer..."):
            result = qa_chain({"query": user_question})
            answer = result['result']  # Ensure the correct field is accessed
            st.write(answer)  # Display the answer

            # Optionally, you can show the source documents if you wish
            if 'source_documents' in result:
                st.write("Source Documents:")
                for doc in result['source_documents']:
                    st.write(doc.page_content)  # or adjust to show parts of the doc
