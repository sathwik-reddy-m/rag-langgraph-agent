from app.retriever import retrieve_documents


if __name__ == "__main__":
    query = "What is LangGraph?"

    docs = retrieve_documents(query)

    print(f"\nQuery: {query}")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content)
