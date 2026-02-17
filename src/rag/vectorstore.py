"""
RAG (Retrieval Augmented Generation) setup for policy documents.
Uses FAISS for vector storage and similarity search.
"""
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
POLICIES_DIR = DATA_DIR / "policies"
VECTORSTORE_PATH = DATA_DIR / "vectorstore"


def load_documents() -> List[Document]:
    """Load all policy documents from the policies directory."""
    if not POLICIES_DIR.exists():
        raise FileNotFoundError(f"Policies directory not found: {POLICIES_DIR}")
    
    documents = []
    
    # Load markdown files
    for file_path in POLICIES_DIR.glob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "type": "policy"
                }
            )
            documents.append(doc)
            print(f"Loaded: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Load text files if any
    for file_path in POLICIES_DIR.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "type": "policy"
                }
            )
            documents.append(doc)
            print(f"Loaded: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents


def create_vectorstore(
    documents: Optional[List[Document]] = None,
    force_rebuild: bool = False
) -> FAISS:
    """
    Create or load the FAISS vectorstore.
    
    Args:
        documents: Optional list of documents. If None, loads from policies dir.
        force_rebuild: If True, rebuilds vectorstore even if it exists.
        
    Returns:
        FAISS vectorstore instance
    """
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Check if vectorstore already exists
    if VECTORSTORE_PATH.exists() and not force_rebuild:
        print("Loading existing vectorstore...")
        return FAISS.load_local(
            str(VECTORSTORE_PATH), 
            embeddings,
            allow_dangerous_deserialization=True
        )
    
    # Load documents if not provided
    if documents is None:
        documents = load_documents()
    
    if not documents:
        raise ValueError("No documents to index")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} document chunks")
    
    # Create vectorstore
    print("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Save for future use
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_PATH))
    print(f"Vectorstore saved to {VECTORSTORE_PATH}")
    
    return vectorstore


def get_retriever(k: int = 3):
    """
    Get a retriever for querying the policy documents.
    
    Args:
        k: Number of documents to retrieve
        
    Returns:
        Retriever instance
    """
    vectorstore = create_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def search_policies(query: str, k: int = 3) -> List[Document]:
    """
    Search policy documents for relevant information.
    
    Args:
        query: Search query
        k: Number of results to return
        
    Returns:
        List of relevant documents
    """
    vectorstore = create_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results


# For testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Build vectorstore
    print("Building vectorstore...")
    vectorstore = create_vectorstore(force_rebuild=True)
    
    # Test search
    print("\nTesting search...")
    results = search_policies("What does Care+ cover?")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} (from {doc.metadata['source']}) ---")
        print(doc.page_content[:300] + "...")
