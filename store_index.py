from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
import pinecone
from langchain.vectorstores import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# 1️⃣ Load and process PDFs
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# 2️⃣ Load embeddings
embeddings = download_hugging_face_embeddings()

# 3️⃣ Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east1-gcp")  # or your AWS region

index_name = "medical-chatbot"

# 4️⃣ Create index if it doesn’t exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine"
    )

# 5️⃣ Connect index to vector store
docsearch = Pinecone.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)