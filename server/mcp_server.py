import os
import sys
import atexit
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# review_store.py lives at project root, one level up from server/. Append it
# to sys.path before importing, same pattern used by agents/retrieval.py and
# agents/risk_scorer.py for their own `from server.mcp_server import ...`.
# Without this, `import review_store` only succeeds by accident (whenever the
# process happens to be launched from project root already).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import review_store

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "qdrant_db")

# Initialize FastMCP server instance
mcp = FastMCP("ClauseGuard-MCP-Server")

# Initialize persistent Qdrant client and embedding model
qdrant_client = QdrantClient(path=DB_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _cleanup_qdrant():
    try:
        qdrant_client.close()
    except Exception:
        pass


atexit.register(_cleanup_qdrant)


def build_policy_collection(collection_name: str, policy_documents: list[dict]):
    """Dynamically creates and populates a session-specific Qdrant collection.

    Args:
        collection_name: Unique session collection name.
        policy_documents: List of dicts containing 'source' and 'text'.
    """
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in collections:
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # chunk_size raised from 500->1200 and overlap from 50->150: at 500 chars,
    # enumerated sub-clauses (e.g. GDPR Art. 17's grounds (a)-(e)) frequently get
    # split across chunk boundaries, so a query only retrieves partial context.
    # The LLM then fills the gap from parametric memory instead of the source
    # text (this is how the fabricated "GDPR 2-year retention" figure happened --
    # that number appears nowhere in gdpr_articles.txt). Larger overlap makes it
    # more likely a full enumerated list lands inside at least one chunk.
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
            vector = embedder.encode(chunk).tolist()
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
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in collections:
        qdrant_client.delete_collection(collection_name)


@mcp.tool()
def search_policy_docs(query_text: str, limit: int = 3, collection_name: str = "company_policies") -> str:
    """Searches internal company compliance policies for text relevant to a query.

    Args:
        query_text: Compliance topic or question.
        limit: Number of top relevant policy passages to return (default: 3).
        collection_name: Specific Qdrant collection to search.

    Returns:
        Formatted string containing matched policy passages and their document sources.
    """
    query_vector = embedder.encode(query_text).tolist()

    try:
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        ).points
    except Exception as e:
        # Prefixed with a distinct sentinel (RETRIEVAL_ERROR::) rather than a
        # plain sentence, so callers -- specifically run_retrieval_agent in
        # retrieval.py -- can reliably detect a failed lookup and stop the
        # clause from being scored, instead of this string silently flowing
        # into the risk-scoring prompt as if it were real policy content.
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


@mcp.tool()
def log_for_human_review(
    contract_name: str,
    clause_text: str,
    risk_level: str,
    justification: str,
    session_id: str = None
) -> str:
    """Logs an evaluated contract clause to the pending review queue for human sign-off.

    Backed by SQLite (see review_store.py) rather than a shared JSON file, so
    concurrent requests can't clobber each other's records, and each record
    gets a stable UUID that later resolve/delete calls reference instead of a
    shifting array index.
    """
    review_id = review_store.add_review(
        contract_name=contract_name,
        clause_text=clause_text,
        risk_level=risk_level,
        justification=justification,
        session_id=session_id,
    )
    return f"Logged clause under '{risk_level.upper()}' risk level for human review (id={review_id})."


if __name__ == "__main__":
    mcp.run()