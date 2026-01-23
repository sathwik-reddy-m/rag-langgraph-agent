from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest_documents():
    # 1. Load raw text
    loader = TextLoader("data/sample.txt")
    documents = loader.load()

    # 2. Convert text -> embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


    # 3. Store embeddings in FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 4. Persist to disk
    vectorstore.save_local("faiss_index")

    print("Documents ingested and vector store created")

