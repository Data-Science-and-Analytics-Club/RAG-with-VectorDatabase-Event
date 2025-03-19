from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
print(HF_TOKEN)

REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(repo_id = REPO_ID, temperature= 0.5, model_kwargs={"token" : HF_TOKEN, "max_length" : 512})
print(llm.invoke("What is Machine Learning"))

CUSTOM_PROMPT_TEMPLATE = """
Use the information in context to answer the user's question.
Strictly stay within the context and do not provide answers to things you do not know.
Do not make up answers. If you do not know the answer, say "I do not know".
Do not provide anything outside the given context.

Context: {context}

Question: {input}

Be extensive and accurate.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "input"])
    return prompt


DB_PATH = "vectorstore/db_faiss"
embedding__model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_PATH, embedding__model, allow_dangerous_deserialization= True)

retriever = db.as_retriever(search_kwargs={ 'k' : 3})

prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

document_chain = create_stuff_documents_chain(llm, prompt_template)

qa_chain = create_retrieval_chain(retriever, document_chain)


def get_response(user_query):
    response = qa_chain.invoke({"input": user_query})
    return response

print(get_response("Ways to deal with anxiety"))