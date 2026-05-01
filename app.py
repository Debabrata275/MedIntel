import os

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

app = FastAPI(title="MedIntel Medical Chatbot")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing. Add it to your .env file.")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = os.getenv("PINECONE_INDEX_NAME", "medicalbot")

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.4,
    max_tokens=500,
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "chat.html")


@app.post("/get", response_class=PlainTextResponse)
async def chat(msg: str = Form(...)):
    print(msg)
    docs = retriever.invoke(msg)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = prompt.format_messages(input=msg, context=context)
    response = llm.invoke(messages)
    answer = response.content
    print("Response : ", answer)
    return answer


@app.get("/health")
async def health():
    return {"status": "ok", "model": GROQ_MODEL, "index": index_name}
