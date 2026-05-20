from app import retriever_store


def retrieve_documents(query: str, k: int = 3):
    """
    Retrieve top-k relevant documents for a query.
    """

    docs_with_scores = retriever_store.vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    return docs_with_scores