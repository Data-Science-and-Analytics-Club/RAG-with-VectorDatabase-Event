import streamlit as st 
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint

import os 
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "vectorstore/db_faiss"
# @st.cache_resource




def main() :
    st.title("Mental Health Chatbot!")

    # if 'messages' not in st.session_state :
    #     st.session_state.messages = []

    # for message in st.session_state.messages :
    #     st.chat_message(message['role']).markdown(message['content'])
    # prompt = st.chat_input("Ask your query here : ")
    # if prompt :
    #     st.chat_message('user').markdown(prompt)
    #     st.session_state.messages.append({'role' : "user", "content" : prompt})




if __name__ == '__main__' :
    main()