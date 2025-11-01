#/tasks.py
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import os, uuid, logging
from celery import Celery
import json
import chromadb
from models import Document_PPI
from utils import chunk_text
from data_extractor import ContextAwarePDFExtractor
from openai import AzureOpenAI
import chromadb.utils as embedding_functions
from database import SessionLocal 
# from llm_call import Embedding_call
from llm_call import AzureEmbeddingFunction,init_azure_client
CELERY_BROKER = os.environ.get("CELERY_BROKER", "redis://127.0.0.1:6379/0")
CELERY_BACKEND = os.environ.get("CELERY_BACKEND1", "redis://127.0.0.1:6379/2")
celery = Celery("ppi_tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)
celery.conf.task_default_queue = "ppi_queue"
celery.conf.result_expires = 3600
celery.conf.task_track_started = True

CHROMA_SERVER = os.environ.get("CHROMA_SERVER", "http://127.0.0.1:8000")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "PPI_collection")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "text-embedding-3-small")
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tasks_PPI")

def get_chroma_collection():
    client = chromadb.HttpClient(host=CHROMA_SERVER)
    emb_fn = AzureEmbeddingFunction()
    return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=emb_fn)

# name="ppi.ingest_document"
@celery.task
def ingest_document_task(documents_filepath:str,file_path: str, filename: str, user_id: int, doc_type: str):
    db = SessionLocal()
    try:
        print(f"Processing {filename} ({doc_type}) for user {user_id}")

        # Save metadata in DB
        doc = Document_PPI(
            user_id=user_id,
            filename=filename,
            file_path=file_path,
            document_type=doc_type
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Extract text 
        extractor = ContextAwarePDFExtractor(file_path)
        data = extractor.extract_all()
        text = data["text_with_context"]

        

        # Chunk
        
        chunks = chunk_text(text)

       
        # Store in Chroma
        collection = get_chroma_collection()
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadatas = [
            {"user_id": user_id, "filename": filename, "document_type": doc_type, "chunk_index": i}
            for i in range(len(chunks))
        ]
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)

        # loading documents question
        with open(documents_filepath, "r") as file:
            questions = json.load(file)

        
        print(f"✅ Ingestion completed for {filename} ({len(chunks)} chunks)")

        client,deployment = init_azure_client()
        answers = []
        for question in questions:
            results = collection.query(
                query_texts=[question],
                n_results=5,
                where={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"document_type": {"$eq": doc_type}}
                    ]
                }
            )

            retrieved_texts = [doc for doc in results.get("documents", [[]])[0]]
            combined_context = "\n".join(retrieved_texts)
         
            # llm call
            prompt = f"""
            Context:
            {combined_context}

            Question: {question}
            Give a concise, factual answer based only on the context.
            """
            stream = client.chat.completions.create(
                model=deployment if deployment else MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=12000,
               
                # temperature=0.0,  # NEW: Deterministic
                stream=False  # NEW: Streaming for partial captures
            )
            final_answer = stream.choices[0].message.content
            answers.append({"Question": question, "Answer": final_answer})
        
        print(answers,"answers")
        return answers
   
        
       
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        raise
    finally:
        db.close()