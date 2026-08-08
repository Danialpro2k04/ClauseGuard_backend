import os
import sys
import atexit
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import cohere

EMBEDDING_MODEL = "embed-english-v3.0"
EMBEDDING_DIM = 1024

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import review_store

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "qdrant_db")

# Initialize FastMCP server instance
mcp = FastMCP("ClauseGuard-MCP-Server")

_qdrant_client = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=DB_PATH)
    return _qdrant_client


def embed_text(text: str, api_key: str, input_type: str = "search_document") -> list[float]:
    """Embeds a single string via Cohere's embed API."""
    client = cohere.Client(api_key)
    response = client.embed(
        model=EMBEDDING_MODEL,
        texts=[text],
        input_type=input_type,
    )
    return response.embeddings[0]


def _cleanup_qdrant():
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass


atexit.register(_cleanup_qdrant)


def build_policy_collection(collection_name: str, policy_documents: list[dict], embedding_api_key: str):
    """Dynamically creates and populates a session-specific Qdrant collection."""
    qdrant_client = get_qdrant_client()

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in collections:
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    points = []
    point_id = 1

    for doc in policy_documents:
        source_name = doc.get("source", "Unknown_Policy")
        content = doc.get("text", "")

        chunks = text_splitter.split_text(content)
        for chunk_idx, chunk in enumerate(chunks):
            vector = embed_text(chunk, embedding_api_key, input_type="search_document")
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": source_name,
                        "chunk_id": chunk_idx
                    }
                )
            )
            point_id += 1

    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)


def drop_policy_collection(collection_name: str):
    """Deletes a session-specific Qdrant collection."""
    qdrant_client = get_qdrant_client()
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in collections:
        qdrant_client.delete_collection(collection_name)


def search_policy_docs(query_text: str, embedding_api_key: str, limit: int = 3, collection_name: str = "company_policies") -> str:
    """Searches internal company compliance policies for text relevant to a query."""
    qdrant_client = get_qdrant_client()

    try:
        query_vector = embed_text(query_text, embedding_api_key, input_type="search_query")
    except Exception as e:
        return f"RETRIEVAL_ERROR:: Embedding request failed: {str(e)}"

    try:
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        ).points
    except Exception as e:
        return f"RETRIEVAL_ERROR:: {str(e)}"

    if not results:
        return "RETRIEVAL_ERROR:: No relevant policy documents found matching the query."

    formatted_passages = []
    for idx, hit in enumerate(results, start=1):
        source = hit.payload.get("source", "Unknown File")
        text = hit.payload.get("text", "")
        formatted_passages.append(
            f"--- [Policy Result {idx}] (Source: {source} | Similarity Score: {hit.score:.2f}) ---\n{text}"
        )

    return "\n\n".join(formatted_passages)


# Register search_policy_docs as an MCP tool while leaving the function callable by agents
mcp.tool()(search_policy_docs)


def log_for_human_review(
    contract_name: str,
    clause_text: str,
    risk_level: str,
    justification: str,
    session_id: str = None
) -> str:
    """Logs an evaluated contract clause to the pending review queue for human sign-off."""
    review_id = review_store.add_review(
        contract_name=contract_name,
        clause_text=clause_text,
        risk_level=risk_level,
        justification=justification,
        session_id=session_id,
    )
    return f"Logged clause under '{risk_level.upper()}' risk level for human review (id={review_id})."


# Register log_for_human_review as an MCP tool
mcp.tool()(log_for_human_review)


if __name__ == "__main__":
    mcp.run()