import os
import uvicorn
import tempfile
import shutil
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline import process_policies, review_contract, delete_session_db
import review_store

app = FastAPI(title="ClauseGuard SaaS API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Normalizes FastAPI/Pydantic validation failures (missing/malformed
    fields, headers, etc.) into a plain {"detail": "<string>"} response.

    By default FastAPI returns {"detail": [{"type", "loc", "msg", "input"}, ...]}
    for these errors. A frontend that expects `detail` to be a string (as this
    app's error-handling generally does) and renders it directly in JSX will
    crash with "Objects are not valid as a React child" the moment one of
    these structured error objects reaches it -- which is exactly what happens
    if a client calls /api/v1/audit without the required Authorization header
    (e.g. an old client still sending api_key as a form field instead of the
    header the endpoint now requires). This handler collapses the structured
    error list into one readable string so the API's error shape stays
    consistent regardless of which validation failed.
    """
    messages = []
    for err in exc.errors():
        loc = " -> ".join(str(part) for part in err.get("loc", []) if part != "body")
        msg = err.get("msg", "Invalid request")
        messages.append(f"{loc}: {msg}" if loc else msg)
    return JSONResponse(
        status_code=422,
        content={"detail": "; ".join(messages) or "Invalid request"},
    )

@app.post("/api/v1/policies")
async def upload_policies(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
    authorization: str = Header(default=None)
):

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Expected 'Authorization: Bearer <embedding_api_key>' header")
    embedding_api_key = authorization.removeprefix("Bearer ").strip()

    temp_dir = tempfile.mkdtemp()
    try:
        saved_files = []
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            saved_files.append(file_path)
            
        process_policies(session_id=session_id, file_paths=saved_files, embedding_api_key=embedding_api_key)
        return {"status": "success", "message": f"Policies embedded for session {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/api/v1/audit")
async def audit_contract(
    session_id: str = Form(...),
    provider: str = Form(...),
    model_name: str = Form(...),
    embedding_api_key: str = Form(...),
    file: UploadFile = File(...),
    authorization: str = Header(default=None)
) -> Dict[str, Any]:
    # api_key now travels as a standard Authorization: Bearer <key> header
    # instead of a multipart Form field. Form fields sit in the same request
    # body as the uploaded file and are more likely to be captured whole by
    # request-logging middleware, reverse proxies, or APM tools that log
    # multipart bodies for debugging. Headers are also logged by some setups,
    # but Authorization is the conventional place secrets go and is far more
    # commonly redacted by default in logging/proxy configs than an arbitrary
    # form field named "api_key".
    #
    # authorization is declared optional (default=None) rather than required,
    # so a missing header falls through to this explicit check and the clean
    # 401 message below -- rather than being rejected earlier by FastAPI's
    # request validation, which would return a structured error object array
    # instead of a plain string message.
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Expected 'Authorization: Bearer <api_key>' header")
    api_key = authorization.removeprefix("Bearer ").strip()

    ext = os.path.splitext(file.filename)[1].lower()
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        report = review_contract(
            file_path=temp_file_path, 
            session_id=session_id,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            embedding_api_key=embedding_api_key,
            original_filename=file.filename
        )
        
        if isinstance(report, dict):
            report["contract_name"] = file.filename
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.delete("/api/v1/session/{session_id}")
async def reset_session(session_id: str):
    delete_session_db(session_id)
    return {"status": "success"}

@app.get("/api/v1/reviews")
async def get_pending_reviews(status: str = None):
    """Fetches the HITL pending review queue from SQLite (see review_store.py).
    Optional ?status=PENDING|RESOLVED query param to filter.
    """
    return review_store.list_reviews(status=status)

@app.delete("/api/v1/reviews/{review_id}")
async def resolve_review(review_id: str):
    """Marks a review resolved, keyed by its stable UUID rather than a shifting
    array index. Index-based deletion was the concurrency bug: two clients
    resolving different reviews at nearly the same time could each read a
    slightly different array ordering and delete the wrong record. A UUID
    primary key makes each resolve action unambiguous regardless of ordering.
    """
    found = review_store.resolve_review(review_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"No pending review found with id={review_id}")
    return {"status": "success"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)