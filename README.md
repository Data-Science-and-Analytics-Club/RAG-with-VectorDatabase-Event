# **Hands-on RAG Implementation with Hugging Face & LangChain**

## **🔍 Overview**

This repository provides a **hands-on implementation** of **Retrieval-Augmented Generation (RAG)** using **Hugging Face LLMs**, **LangChain**, and **FAISS vector databases**. The project demonstrates how to build an **AI-powered chatbot** capable of retrieving relevant context from documents and generating responses using **Mistral-7B**. 🚀

## **📌 Features**

- **Document Processing**: Load and chunk PDF documents for retrieval.
- **Vector Database (FAISS)**: Store embeddings for efficient similarity search.
- **Hugging Face LLM Integration**: Use **Mistral-7B** for intelligent responses.
- **Custom Prompt Engineering**: Ensure accurate, context-aware answers.
- **Interactive Chatbot UI**: Built with **Streamlit** for easy interaction.

## **🛠 Prerequisites**

Ensure you have the following installed before proceeding:

- Python 3.8+
- pip
- Hugging Face account & API Token
- [Streamlit](https://streamlit.io)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://python.langchain.com/)

## **📦 Installation**

Clone the repository and install dependencies:

```bash
git clone https://github.com/Data-Science-and-Analytics-Club/RAG-with-VectorDatabase-Event.git
cd RAG-with-VectorDatabase-Event
pip install -r requirements.txt
```

Set up your **Hugging Face API token**:

```bash
export HF_TOKEN='your_huggingface_api_token'
```

---

## **1️⃣ Context Creation & Embedding Generation**

### **🔹 Necessary Imports**

First, we need to import the required modules for document processing, text chunking, and embedding generation.

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
```

### **🔹 Function Explanation**

#### **Loading PDF Documents**

The `load_pdfs` function loads PDF files from the specified directory using **LangChain's DirectoryLoader** and **PyPDFLoader**.

```python
def load_pdfs(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()
```

#### **Creating Chunks from Documents**

Since LLMs work better with smaller context windows, we split documents into **manageable chunks** using `RecursiveCharacterTextSplitter`.

```python
def create_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(data)
```

#### **Generating Embeddings**

Embeddings help in semantic search and retrieval. We use **Hugging Face's MiniLM model** to generate vector embeddings.

```python
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### **🔹 Bringing It All Together**

Finally, we load the documents, create text chunks, generate embeddings, and store them in a **FAISS vector database**.

```python
documents = load_pdfs("context/")
text_chunks = create_chunks(documents)
embedding_model = get_embedding_model()

DB_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_PATH)
```

---

## **🚀 Next Steps: Memory & Retrieval**

Next, we will cover how to **load an LLM and retrieve relevant document chunks for answering user queries!** 🎯

## Overview

This project implements a retrieval-augmented generation (RAG) system using `LangChain` and `HuggingFace`. It leverages a FAISS vector store for efficient document retrieval and a Hugging Face hosted model (`Mistral-7B-Instruct-v0.3`) for generating responses. The pipeline follows these key steps:

1. **Import necessary libraries**
2. **Load environment variables**
3. **Initialize the Hugging Face model endpoint**
4. **Define a custom prompt template**
5. **Load a FAISS vector store for document retrieval**
6. **Create retrieval and document processing chains**
7. **Build the complete retrieval-augmented pipeline**
8. **Define a function to handle user queries**

---

## Step 1: Import Necessary Libraries

To set up our pipeline, we import the required libraries:

```python
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
```

These libraries enable:

- **LLM integration** (`HuggingFaceEndpoint`)
- **Prompt customization** (`PromptTemplate`)
- **Retrieval-augmented QA pipeline** (`RetrievalQA`)
- **Document combination strategies** (`create_stuff_documents_chain`)
- **Vector store retrieval** (`FAISS`)
- **Environment variable handling** (`dotenv`)

---

## Step 2: Load Environment Variables

We use `dotenv` to load sensitive credentials, such as the Hugging Face API token:

```python
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
```

Ensure you have a `.env` file with:

```
HF_TOKEN=your_huggingface_api_key
```

---

## Step 3: Load the Hugging Face Model Endpoint

We define a function to initialize the `HuggingFaceEndpoint` with the required parameters:

```python
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm
```

- `temperature=0.5`: Balances randomness and determinism in responses.
- `max_length=512`: Limits the response length.

---

## Step 4: Define a Custom Prompt Template

A prompt template ensures that responses stay within the provided context:

```python
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
```

---

## Step 5: Load FAISS Vector Store

We load a pre-built FAISS vector store, which contains embeddings of documents for efficient retrieval:

```python
DB_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
```

- `sentence-transformers/all-MiniLM-L6-v2`: Embedding model used to encode text into vector space.
- `FAISS.load_local()`: Loads the vector store from disk.

---

## Step 6: Create the Retriever

The retriever fetches relevant documents based on the query:

```python
retriever = db.as_retriever(search_kwargs={'k': 3})
```

- `k=3`: Retrieves the top 3 most relevant documents.

---

## Step 7: Create the QA Document Chain

We initialize the LLM and prompt, then create the document processing chain:

```python
llm = load_llm(HUGGINGFACE_REPO_ID)
prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
qa_document_chain = create_stuff_documents_chain(llm, prompt_template)
```

---

## Step 8: Create the Retrieval Chain

We combine the retriever and document processing chain into a complete retrieval-augmented generation (RAG) pipeline:

```python
qa_chain = create_retrieval_chain(retriever, qa_document_chain)
```

---

## Step 9: Define the Response Function

This function processes user queries and returns an answer:

```python
def get_response(user_query):
    response = qa_chain.invoke({"input": user_query})
    return response['answer']
```

### Example Usage

```python
user_query = "What is LangChain?"
response = get_response(user_query)
print(response)
```

---

## In Short

This project implements a RAG-based chatbot using:

- **FAISS for document retrieval**
- **Hugging Face models for text generation**
- **LangChain for pipeline management**
- **A structured prompt template for better responses**

To run the project:

1. Install dependencies (`pip install langchain faiss-cpu transformers sentence-transformers`)
2. Add your Hugging Face token to `.env`
3. Load the FAISS vector store with relevant data
4. Call `get_response(user_query)` to get an AI-generated response.
