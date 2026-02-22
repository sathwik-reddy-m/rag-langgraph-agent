from langchain_groq import ChatGroq

def generate_answer(query: str, documents: list[str]) -> str:
    """
    Generate an answer using retrieved documents as context.
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    context = "\n\n".join(documents)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""
    
    response = llm.invoke(prompt)
    return response.content

