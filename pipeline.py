import os
import pypdf
from docx import Document

from agents.intake import run_intake_agent
from agents.retrieval import run_retrieval_agent
from agents.risk_scorer import run_risk_scoring_agent
from server.mcp_server import build_policy_collection, drop_policy_collection

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        extracted_text = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(extracted_text)
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    else:
        raise ValueError(f"Unsupported format: '{ext}'")

def process_policies(session_id: str, file_paths: list[str], embedding_api_key: str):
    print(f"  Building Policy Database for Session: {session_id}")
    policy_documents = []
    for path in file_paths:
        text_content = extract_text_from_file(path)
        filename = os.path.basename(path)
        policy_documents.append({"source": filename, "text": text_content})

    build_policy_collection(collection_name=session_id, policy_documents=policy_documents, embedding_api_key=embedding_api_key)
    return {"status": "success"}

def review_contract(file_path: str, session_id: str, provider: str, model_name: str, api_key: str, embedding_api_key: str, top_k_policies: int = 4, original_filename: str | None = None) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Contract not found: {file_path}")

    contract_name = original_filename or os.path.basename(file_path)
    contract_text = extract_text_from_file(file_path)

    print(" [Step 1/3] Running Intake Agent...")
    intake_data = run_intake_agent(contract_text, provider, model_name, api_key)

    print("\n [Step 2/3] Running Retrieval Agent...")
    retrieval_data = run_retrieval_agent(intake_data, embedding_api_key=embedding_api_key, collection_name=session_id, top_k=top_k_policies)

    print("\n [Step 3/3] Running Risk-Scoring Agent...")
    final_report = run_risk_scoring_agent(retrieval_data, contract_name, provider, model_name, api_key, session_id=session_id)

    return final_report

def delete_session_db(session_id: str):
    drop_policy_collection(collection_name=session_id)