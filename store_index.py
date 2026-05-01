from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
import os
import time


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing. Add it to your .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


print("Loading PDF files from Data/...")
extracted_data = load_pdf_file(data='Data/')
print(f"Loaded {len(extracted_data)} PDF pages/documents.")

print("Splitting PDF text into chunks...")
text_chunks = text_split(extracted_data)
print(f"Created {len(text_chunks)} text chunks.")

print("Loading Hugging Face embedding model...")
embeddings = download_hugging_face_embeddings()
print("Embedding model is ready.")


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = os.getenv("PINECONE_INDEX_NAME", "medicalbot")


if not pc.has_index(index_name):
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_REGION", "us-east-1")
        )
    )

    while True:
        status = pc.describe_index(index_name).status
        ready = status["ready"] if isinstance(status, dict) else status.ready
        if ready:
            break
        print("Waiting for Pinecone index to be ready...")
        time.sleep(1)
else:
    print(f"Using existing Pinecone index '{index_name}'.")

print("Embedding chunks and uploading them to Pinecone. This can take a few minutes...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Done. Embeddings have been stored in Pinecone.")
