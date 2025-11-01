
# backend/main.py
import os, uuid, shutil
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db, Base, engine
from models import User, Chat, Document
from auth import hash_password, verify_password, create_access_token, decode_access_token, oauth2_scheme
from tasks import ingest_document,get_chroma_collection
import chromadb
from chromadb.utils import embedding_functions
from utils import chunk_text,merge_to_excel_and_word
# from llm_call import Embedding_call,init_azure_client
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import Form
from llm_call import AzureEmbeddingFunction,init_azure_client
import time
from fastapi.responses import JSONResponse
from typing import Optional,List
from tasks_PPI import ingest_document_task
from celery.result import AsyncResult
from celery import Celery
from tasks_PPI import celery as celery_app,ingest_document_task
from datetime import datetime
# register manually
celery_app.tasks.register(ingest_document_task)

from fastapi.staticfiles import StaticFiles
# CELERY_BROKER = os.environ.get("CELERY_BROKER", "redis://127.0.0.1:6379/0")
# CELERY_BACKEND = os.environ.get("CELERY_BACKEND", "redis://127.0.0.1:6379/1")
# celery = Celery("tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    # hash_password:str

class UserLogin(BaseModel):
    username: str
    password: str


class FinalReportRequest(BaseModel):
    task_ids: List[str] 


Base.metadata.create_all(bind=engine)
load_dotenv()
app = FastAPI(title="RAG Chat Application")

deployment = os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS")

RFP_PATH = os.getenv("RFP_PATH")
OTHER_DOCUMENTS_PATH = os.getenv("OTHER_DOCUMENTS_PATH")
RESPONSE_PATH = os.getenv("RESPONSE_PATH")

origins = [
    "http://127.0.0.1:5500",  # your frontend origin
    "http://localhost:5500",  # just in case
    "http://localhost:8001", 
    "http://localhost:8002" # optional
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR","./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# mounting the frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
# ------------------------
# Auth endpoints
# ------------------------
# MAX_BCRYPT_LENGTH = 70


@app.post("/signup")
def signup(user: UserCreate, db:Session=Depends(get_db)):
    if db.query(User).filter(User.username==user.username).first():
        raise HTTPException(400,"Username exists")
    if db.query(User).filter(User.email==user.email).first():
        raise HTTPException(400,"Email exists")
    hashed_pw = hash_password(user.password)
    user = User(username=user.username, email=user.email, hashed_password=hashed_pw)
   
    db.add(user)
   

    db.commit()
    db.refresh(user)
    
    token = create_access_token({"sub":user.username,"id":user.id})
    return {"access_token":token,"token_type":"bearer"}

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
   
    db_user = db.query(User).filter(User.username == user.username).first()
    
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": db_user.username, "id": db_user.id})
    return {"access_token": token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    print("TOKEN RECEIVED:", token)
    try:
        payload = decode_access_token(token)
        print("PAYLOAD:", payload)
    except Exception as e:
        print("DECODE ERROR:", e)
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == payload.get("id")).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ------------------------
# Upload documents
# ------------------------
@app.post("/upload")
def upload(file:UploadFile=File(...), current_user:User=Depends(get_current_user), db:Session=Depends(get_db)):
    filename = file.filename
    print(filename,"filename")
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")
    with open(file_path,"wb") as f:
        shutil.copyfileobj(file.file, f)
    doc = Document(user_id=current_user.id, filename=filename, file_path=file_path)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    ingest_document.delay(file_path, filename, current_user.id)
    return {"status":"uploaded","document_id":doc.id,"filename":filename}

# ------------------------
# Query RAG
# ------------------------
@app.post("/query")
def query(req: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # query chroma for current_user documents only
    question = req.get("question", "").strip()
    print(question,"question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    # static top_k
    top_k = 5  # static value

    client = chromadb.HttpClient(host=os.environ.get("CHROMA_SERVER","http://chroma:8000"))
    emb_fn = AzureEmbeddingFunction()
    collection = client.get_or_create_collection(name=os.environ.get("CHROMA_COLLECTION","rag_collection"), embedding_function=emb_fn)
    qres = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents","metadatas"]
    )
    docs = qres["documents"][0]
    metas = qres["metadatas"][0]
    # filter by user_id
    snippets=[]
    for d,m in zip(docs, metas):
        if m.get("user_id")==current_user.id:
            snippets.append(d)
    answer="\n\n".join(snippets)
    # llm call
    client,deployment = init_azure_client()
    prompt = f"""
            You are a helpful assistant. Answer the following briefly with full understanding of question and the provided context.
            always give something in answer that is related to the question and context.
            Question: {question}

            Context:
            {answer}
            """
    stream = client.chat.completions.create(
                model=deployment if deployment else MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
                # top_p=1.0,
                # temperature = 0.0,
                # frequency_penalty = 0.0,
                # presence_penalty = 0.0,
                  # Use dynamic max
                # temperature=0.0,  # NEW: Deterministic
                stream=False  # NEW: Streaming for partial captures
            )
    final_answer = stream.choices[0].message.content

     


    # save chat history
    chat = Chat(user_id=current_user.id, question=question, answer=final_answer)
    db:Session = next(get_db())
    db.add(chat)
    db.commit()
    
    # print(final_answer,"final_answer")
    print({"answer":final_answer,"sources":snippets})
    # time.sleep(10) 
    # print(answer,"answer")
    return JSONResponse(content={"answer": final_answer, "sources": snippets})

# ------------------------
# Chat history
# ------------------------
@app.get("/history")
def history(current_user:User=Depends(get_current_user), db:Session=Depends(get_db)):
    chats = db.query(Chat).filter(Chat.user_id==current_user.id).order_by(Chat.created_at.desc()).all()
    return [{"question":c.question,"answer":c.answer,"created_at":c.created_at} for c in chats]


# getting the count of the documents
@app.get("/document_count")
def document_count(current_user:User=Depends(get_current_user), db:Session=Depends(get_db)):
    coll = get_chroma_collection()
    total_docs = coll.count()
    print(f"Total documents (chunks) stored: {total_docs}")
    return total_docs

# getting the documents names 
@app.get("/documents")
def list_user_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Returns all uploaded documents for the current user along with chunk count.
    """
    coll = get_chroma_collection()
    results = coll.get(include=["metadatas"])

    user_files = {}

    if results and results.get("metadatas"):
        for meta in results["metadatas"]:
            if meta.get("user_id") != current_user.id:
                continue
            fname = meta.get("filename", "Unknown")
            if fname not in user_files:
                user_files[fname] = {"filename": fname, "chunk_count": 0}
            user_files[fname]["chunk_count"] += 1

    # Also include file info from your database (Document table)
    db_docs = db.query(Document).filter(Document.user_id == current_user.id).all()
    db_files = []
    for doc in db_docs:
        db_files.append({
            "filename": doc.filename,
            "file_path": doc.file_path,
            "created_at": doc.uploaded_at,
            "chunks": user_files.get(doc.filename, {}).get("chunk_count", 0)
        })

    return {"documents": db_files}


# deleting the documents
@app.delete("/documents/{filename}")
def delete_user_document(filename: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Deletes all chunks of a given file for the current user from ChromaDB
    and removes the file entry from the database.
    """
    coll = get_chroma_collection()
    # results = coll.get(include=["ids", "metadatas"])
    results = coll.get(include=["metadatas"])

    if not results or not results.get("metadatas"):
        raise HTTPException(status_code=404, detail="No documents found in ChromaDB")

    to_delete = []
    for idx, meta in enumerate(results["metadatas"]):
        if meta.get("user_id") == current_user.id and meta.get("filename") == filename:
            to_delete.append(results["ids"][idx])

    if not to_delete:
        raise HTTPException(status_code=404, detail=f"No chunks found for '{filename}'")

    # Delete from ChromaDB
    coll.delete(ids=to_delete)

    # Delete from database if exists
    db_doc = db.query(Document).filter(
        Document.user_id == current_user.id,
        Document.filename == filename
    ).first()
    if db_doc:
        db.delete(db_doc)
        db.commit()

        # Delete physical file if you wish
        if os.path.exists(db_doc.file_path):
            os.remove(db_doc.file_path)

    return {"status": "deleted", "filename": filename, "deleted_chunks": len(to_delete)}


# uploading documents
# print("CELERY BACKEND:", celery_app.conf.result_backend)
@app.post("/upload_documents")
async def upload_documents(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    rfp: Optional[UploadFile] = None,
    gtp: Optional[UploadFile] = None,
    response: Optional[UploadFile] = None,
    other: Optional[UploadFile] = None,
    db: Session = Depends(get_db)
):
    """
    Upload 1–4 documents asynchronously.
    Each document triggers a Celery ingestion & Q/A task.
    """
    uploaded_files = {
        "RFP": rfp,
        "GTP": gtp,
        "Response": response,
        "Other": other
    }

    tasks = []
    for doc_type, file in uploaded_files.items():
        if not file:
            continue

        filename = file.filename
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"📁 Received {filename} ({doc_type}) from user {current_user}")

        # Celery async task trigger
        print("reached")
        if doc_type == "RFP":
            task = ingest_document_task.delay(RFP_PATH,file_path, filename, current_user.id, doc_type)
            tasks.append((doc_type, task.id))
            print("RFP task completed",task.id)
        elif doc_type == "Other":
            task = ingest_document_task.delay(OTHER_DOCUMENTS_PATH,file_path, filename, current_user.id, doc_type)
            tasks.append((doc_type, task.id))
            print("other task completed")
        else:
            task = ingest_document_task.delay(RESPONSE_PATH,file_path, filename, current_user.id, doc_type)
            tasks.append((doc_type, task.id))
            print("response task completed")

    if not tasks:
        raise HTTPException(status_code=400, detail="No files uploaded")
    print("reached there")
    print(tasks,"tasks")
    return JSONResponse(
        {"message": "✅ Upload successful. Processing started.",
         "task_ids": tasks}
    )


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    try:
        task = celery_app.AsyncResult(task_id)
        state = task.state
        print(state,"state")
       
        result = task.result if state == "SUCCESS" else None
        return {"task_id": task_id, "state": state, "status": state, "result": result}
    except Exception as e:
        print("error")
        return {"task_id": task_id, "state": "UNKNOWN", "status": "UNKNOWN", "error": str(e)}


@app.post("/generate_final_report")
async def generate_final_report(req: FinalReportRequest):
    if not req or not req.task_ids:
        raise HTTPException(status_code=400, detail="No task IDs provided")

    all_answers = []
    pending_tasks = []

    for task_id in req.task_ids:
        result = celery_app.AsyncResult(task_id)  # use same app
        print(f"check {task_id} -> {result.state}")
        if result.state == "PENDING":
            pending_tasks.append(task_id)
        elif result.state == "FAILURE":
            raise HTTPException(status_code=500, detail=f"Task {task_id} failed: {result.result}")
        elif result.state == "SUCCESS":
            # ensure result is serializable and safe
            all_answers.extend(result.result or [])
        else:
            pending_tasks.append(task_id)

    if pending_tasks:
        raise HTTPException(status_code=400, detail=f"Tasks not completed yet: {', '.join(pending_tasks)}")

    if not all_answers:
        raise HTTPException(status_code=400, detail="No answers found to generate report")

    output_dir = os.path.join("reports", datetime.now().strftime("%Y%m%d"))
    os.makedirs(output_dir, exist_ok=True)
    excel_path, word_path = merge_to_excel_and_word(all_answers,output_dir)

    return JSONResponse({
        "message": "✅ Report generated successfully",
        "excel_path": excel_path,
        "word_path": word_path
    })

@app.get("/download/{filename}")
async def download_file(filename: str):
    print("reached download")
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)


@app.post("/cancel_task/{task_id}")
def cancel_task(task_id: str):
    """Cancel a running or pending Celery task."""
    try:
        task = AsyncResult(task_id, app=celery_app)
        if task.state in ["SUCCESS", "FAILURE", "REVOKED"]:
            return {"message": f"Task {task_id} already {task.state}"}

        task.revoke(terminate=True)
        return {"message": f"Task {task_id} cancelled successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



