# helper.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ fixed
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document  # ✅ fixed

# -----------------------------
# 1️⃣ Extract Data From PDF
# -----------------------------
def load_pdf_file(data: str) -> List[Document]:
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# -----------------------------
# 2️⃣ Filter to Minimal Docs
# -----------------------------
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only 'source' metadata and page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# -----------------------------
# 3️⃣ Split the Data into Chunks
# -----------------------------
def text_split(extracted_data: List[Document], chunk_size: int = 500, chunk_overlap: int = 20) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# -----------------------------
# 4️⃣ Load HuggingFace Embeddings
# -----------------------------
def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'  # 384-dimensional
    )
    return embeddings