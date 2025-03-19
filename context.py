from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_pdfs(path) :
    loader = DirectoryLoader( path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def create_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(data)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

DB_PATH = "vectorstore/db_faiss"

documents = load_pdfs("CONTEXT/")
print(len(documents))
text_chunks  = create_chunks(documents)
print(len(text_chunks))
embedding_model = get_embedding_model()
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_PATH)