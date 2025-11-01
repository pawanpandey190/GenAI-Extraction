# backend/tasks.py
import os, uuid, logging
from celery import Celery
import chromadb
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from utils import extract_text_from_file, chunk_text
from openai import AzureOpenAI
import chromadb.utils as embedding_functions
# from llm_call import Embedding_call
from llm_call import AzureEmbeddingFunction
CELERY_BROKER = os.environ.get("CELERY_BROKER", "redis://127.0.0.1:6379/0")
CELERY_BACKEND = os.environ.get("CELERY_BACKEND", "redis://127.0.0.1:6379/1")
celery = Celery("rag_tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)
celery.conf.task_default_queue = "rag_queue"
CHROMA_SERVER = os.environ.get("CHROMA_SERVER", "http://127.0.0.1:8000")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "rag_collection")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "text-embedding-3-small")
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tasks")

def get_chroma_collection():
    client = chromadb.HttpClient(host=CHROMA_SERVER)
    emb_fn = AzureEmbeddingFunction()
    return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=emb_fn)

@celery.task(name="ingest_document")
def ingest_document(file_path: str, filename: str, user_id: int):
    logger.info("Ingesting %s", filename)
    text = extract_text_from_file(filename, open(file_path,"rb").read())
    chunks = chunk_text(text)
    coll = get_chroma_collection()
    ids, docs, metas = [], [], []
    for i, c in enumerate(chunks):
        uid = str(uuid.uuid4())
        ids.append(uid)
        docs.append(c)
        metas.append({"source_id": uid, "user_id": user_id, "filename": filename, "chunk_index": i})
    coll.add(documents=docs, metadatas=metas, ids=ids)
    return {"status":"ok","chunks":len(chunks)}
