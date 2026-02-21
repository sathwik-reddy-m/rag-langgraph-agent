from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def retrieve_documents(query: str, k: int = 3):
    """
    Retrieve top-k relevant documents for a query.
    """

    # 1. Load embeddings model (must be SAME as ingestion)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Load FAISS index from disk
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # 3. Perform similarity search
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    return docs_with_scores
