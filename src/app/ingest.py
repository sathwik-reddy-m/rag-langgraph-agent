from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


def ingest_documents():
    # 1. Load raw document
    loader = TextLoader("data/sample.txt")
    documents = loader.load()

    # 2. Create embeddings model (same everywhere)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Semantic chunking
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=30,
    )

    chunks = splitter.split_documents(documents)

    # 4. Store chunks in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Persist to disk
    vectorstore.save_local("faiss_index")

    print(f"✅ Ingested {len(chunks)} semantic chunks into vector store")
