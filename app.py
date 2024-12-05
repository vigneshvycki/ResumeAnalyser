import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title('Data Science')
llm=ChatGroq(groq_api_key=groq_api_key,
             model='Llama3-8b-8192')

prompt=ChatPromptTemplate.from_template(
    '''Answer the question based on the provided content only.
    Please provide the most accurate response based on the question.
    If the question is about HOD the first person will be the HOD.
    <context>
    {context}
    <context>
    Question:{input}
    '''
)

if 'vector' not in st.session_state:
    st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    st.session_state.loader=WebBaseLoader(['https://www.sjctni.edu/Department/hishome.jsp?deptCode=DS&id=1&bredcom=Home%20|%20Academics%20|%20Departments%20|%20DATA%20SCIENCE',
                                        'https://www.sjctni.edu/Department/hishome.jsp?deptCode=DS&id=23&bredcom=Home%20|%20Academics%20|%20Schools%20|%20School%20of%20Computing%20Sciences%20|%20DATA%20SCIENCE%20|%20Programme%20Specific%20Outcomes',
                                        'https://www.sjctni.edu/Department/hishome.jsp?deptCode=DS&id=10&bredcom=Home%20|%20Academics%20|%20Schools%20|%20School%20of%20Computing%20Sciences%20|%20DATA%20SCIENCE%20|%20Faculty',
                                        ''])
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_spliter.split_documents(st.session_state.docs[:100])
    st.session_state.vector=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    
document_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.vector.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)

prompt=st.text_input('Type you input here:')

if prompt:
    response=retrieval_chain.invoke({'input':prompt})
    st.write(response['answer'])
