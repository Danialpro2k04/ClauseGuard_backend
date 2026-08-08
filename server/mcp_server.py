import os
import sys
import atexit
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import cohere

# Embeddings are produced via Cohere's embed API instead of a locally-loaded
# model. Cohere's free Trial API key (1,000 calls/month, no credit card
# required, issued automatically on signup) is currently the most reliable
# free option among hosted embedding providers -- OpenAI discontinued its
# automatic signup credit and now requires prepaid credits for anything
# beyond a very limited free chat model, which doesn't cover embeddings.
# Using a hosted API (rather than a local model) also removes torch from the
# dependency tree entirely, so there's no local memory cost for embedding.
EMBEDDING_MODEL = "embed-english-v3.0"
EMBEDDING_DIM = 1024

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

# Qdrant's client is still process-local and lazy (created once, reused).
# The embedding client is now a thin OpenAI API wrapper rather than a loaded
# model, so there's nothing heavy to defer -- but it's still created per-call
# (not cached globally) since each user supplies their own EMBEDDING_API_KEY,
# and different sessions/requests may use different keys.
_qdrant_client = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=DB_PATH)
    return _qdrant_client


def embed_text(text: str, api_key: str, input_type: str = "search_document") -> list[float]:
    """Embeds a single string via Cohere's embed API.

    Args:
        text: The text to embed.
        api_key: The user-supplied EMBEDDING_API_KEY (a Cohere API key).
        input_type: "search_document" for policy text being stored, or
            "search_query" for a query being searched against it -- Cohere's
            embed models are tuned differently for each, which improves
            retrieval quality when queries and documents are labeled
            correctly rather than always using the same input_type.

    Returns:
        A 1024-dimension embedding vector.
    """
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
    """Dynamically creates and populates a session-specific Qdrant collection.

    Args:
        collection_name: Unique session collection name.
        policy_documents: List of dicts containing 'source' and 'text'.
        embedding_api_key: User-supplied Cohere API key used to generate embeddings.
    """
    qdrant_client = get_qdrant_client()

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in collections:
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
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


@mcp.tool()
def search_policy_docs(query_text: str, embedding_api_key: str, limit: int = 3, collection_name: str = "company_policies") -> str:
    """Searches internal company compliance policies for text relevant to a query.

    Args:
        query_text: Compliance topic or question.
        embedding_api_key: User-supplied Cohere API key used to generate the query embedding.
        limit: Number of top relevant policy passages to return (default: 3).
        collection_name: Specific Qdrant collection to search.

    Returns:
        Formatted string containing matched policy passages and their document sources.
    """
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