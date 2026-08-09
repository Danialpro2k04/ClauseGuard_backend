import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.mcp_server import search_policy_docs


def _merge_policy_contexts(context_a: str, context_b: str) -> str:
    """Merges two formatted policy-passage strings from search_policy_docs,
    deduplicating passages that appear in both (matched on their exact text
    block, since each passage block already embeds its own source/score header).
    Order is preserved with context_a's passages first.
    """
    if not context_a or context_a.startswith("RETRIEVAL_ERROR::"):
        return context_b
    if not context_b or context_b.startswith("RETRIEVAL_ERROR::"):
        return context_a

    blocks_a = context_a.split("\n\n")
    blocks_b = context_b.split("\n\n")

    seen = set(blocks_a)
    merged = list(blocks_a)
    for block in blocks_b:
        if block not in seen:
            merged.append(block)
            seen.add(block)

    return "\n\n".join(merged)


def run_retrieval_agent(intake_data: dict, embedding_api_key: str, collection_name: str = "company_policies", top_k: int = 2) -> dict:
    evaluated_clauses = []

    clauses = intake_data.get("clauses", [])
    print(f"🔍 Retrieval Agent processing {len(clauses)} clause(s) against collection '{collection_name}'...")

    for idx, clause in enumerate(clauses, start=1):
        statement = clause.get("compliance_statement", "")
        clause_text = clause.get("clause_text", "")
        print(f"  └─ [{idx}/{len(clauses)}] Querying policy database for: '{statement[:60]}...'")

        retrieved_context = search_policy_docs(
            query_text=clause_text,
            embedding_api_key=embedding_api_key,
            limit=top_k,
            collection_name=collection_name
        )
        if statement:
            statement_context = search_policy_docs(
                query_text=statement,
                embedding_api_key=embedding_api_key,
                limit=top_k,
                collection_name=collection_name
            )
            retrieved_context = _merge_policy_contexts(retrieved_context, statement_context)

        retrieval_failed = retrieved_context.startswith("RETRIEVAL_ERROR::")

        evaluated_clauses.append({
            "clause_title": clause.get("clause_title", "Untitled Clause"),
            "clause_text": clause.get("clause_text", ""),
            "compliance_statement": statement,
            "retrieved_policy_context": retrieved_context,
            "retrieval_failed": retrieval_failed
        })

    return {
        "document_type": intake_data.get("document_type", "Unknown"),
        "clauses": evaluated_clauses
    }