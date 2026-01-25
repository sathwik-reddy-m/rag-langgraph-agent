from app.retriever import retrieve_documents
from app.generator import generate_answer

if __name__ == "__main__":
    query= "What is LangGraph"

    docs = retrieve_documents(query)
    contents = [doc.page_content for doc in docs]

    answer = generate_answer(query, contents)

    print("\nQuestion:", query)
    print("\nAnswer:", answer)

