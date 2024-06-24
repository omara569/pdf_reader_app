import streamlit as st
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # for google's generative AI embeddings
from langchain_community.vectorstores import FAISS # better for faster service
from langchain.chains.question_answering import load_qa_chain # helps us with chatting and prompt definition
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil

load_dotenv() # load the environment variables unique to this app
api_key = os.getenv('GEM_API_KEY') 


# we're going to create an application that takes PDFs and converts them into vector embeddings. Afterwards, we can ask the LLM questions on the PDFs and retrieve responses accordingly
def extract_pdf_text(pdf_files: str):
    text = ''
    for pdf_file in pdf_files:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def chunkize_text(text:str, chunk_size=10000, chunk_overlap=1000):
    """
        Takes the text and partitions it into chunks
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) # divide into pieces of characters, with some overlap
    chunks = splitter.split_text(text)
    return chunks


def store_embeddings(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key) # This is how we'll create our embedding. The embedding model we're using is the first one
    if os.path.exists('faiss_index'):
        shutil.rmtree('faiss_index')
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local('faiss_index')

def conversational_chain(api_key):
    prompt = """Answer the query with as much detail as possible using the provided context. If the requested information is unavailable, simply indicate that such is the case:
        Context: \n{context}\n
        Query: \n{query}\n

        Response:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=.5, google_api_key=api_key) # get's our gemini model and it's output variation
    prompt = PromptTemplate(template = prompt, input_variables=['context', 'query']) # create prompt for the LLM
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt) # we use the "stuff" chain type because we're stuffing both the context and the query into one prompt. Suitable for smaller scale tasks

    return chain

def user_input(user_text: str, api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key) # select the embedding of the characters
    vectorstore = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True) # load the vector
    results = vectorstore.similarity_search(user_text)

    chain = conversational_chain(api_key)

    response = chain.run({'input_documents': results, 'query': user_text})
    return response


# App Design
st.title('PDF chatting using Gemini and Langchain')
st.write('The following application allows a user to upload multiple PDFs and chat with Gemini pro on the contents of the PDFs. Note that this application is memoryless - uploading new files deletes the previous ones.')

text_input = st.text_input('Input queries on the PDF files here:')
query_proc = st.button('Process Query')
query_response = st.empty()

uploaded_files = st.sidebar.file_uploader(label='Upload PDF Files Here', type=['pdf'] , accept_multiple_files=True)
process_files = st.sidebar.button(label='Process PDFs')
text_content = ''


# callback functionality
if uploaded_files and process_files:
    # extract text from the PDF
    query_response.write('Loading PDFs . . . may take some time')
    text_content = extract_pdf_text(uploaded_files)
    text_chunks = chunkize_text(text_content)
    store_embeddings(text_chunks, api_key=api_key)
    query_response.write('Done loading PDFs. Enter query')
    
if text_input and query_proc and uploaded_files:
    response = user_input(text_input, api_key=api_key)
    query_response.markdown(response)
elif text_input and query_proc and (not uploaded_files):
    query_response.write('Files not uploaded!')
elif uploaded_files and query_proc and (not text_input):
    query_response.write('No query entered!')