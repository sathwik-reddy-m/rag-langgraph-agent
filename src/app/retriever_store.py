from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Global shared resources
embeddings = None
vectorstore = None


def initialize_retriever():
    """
    Initialize embedding model and vector store once at startup.
    """

    global embeddings
    global vectorstore

    print("Initializing embeddings model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading FAISS index...")

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("Retriever initialization complete.")