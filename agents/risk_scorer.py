import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.mcp_server import log_for_human_review
from agents.llm import call_llm

def score_clause_risk(contract_name: str, clause_info: dict, provider: str, model_name: str, api_key: str, session_id: str = None) -> dict:
    """Evaluates a single contract clause against retrieved company policy context."""
    clause_title = clause_info.get("clause_title", "Untitled Clause")
    clause_text = clause_info.get("clause_text", "")
    policy_context = clause_info.get("retrieved_policy_context", "")

    # If retrieval itself failed (Qdrant error, or genuinely no matching policy
    # found), don't send that error string to the LLM as if it were real policy
    # content -- it can't ground a risk judgment in text that isn't policy, and
    # doing so risks the model rationalizing a score from nothing. Instead route
    # straight to human review as UNVERIFIED so a person checks it, rather than
    # letting the pipeline silently emit a confident-looking but ungrounded score.
    if clause_info.get("retrieval_failed"):
        justification = (
            "Automated policy retrieval failed or found no matching policy content "
            f"for this clause. Raw retrieval result: {policy_context}"
        )
        log_msg = log_for_human_review(
            contract_name=contract_name,
            clause_text=clause_text,
            risk_level="UNVERIFIED",
            justification=justification,
            session_id=session_id
        )
        print(f"MCP HITL Tool: {log_msg}")
        return {
            "clause_title": clause_title,
            "clause_text": clause_text,
            "risk_level": "UNVERIFIED",
            "justification": justification,
            "recommendation": "Manually review against company policy; automated retrieval could not locate relevant policy text."
        }

    system_prompt = (
        "You are an expert Corporate Compliance Risk Assessor. Your task is to compare a "
        "proposed contract clause against retrieved internal company compliance policies.\n\n"
        "Assign one of the following risk levels:\n"
        "- HIGH: Direct violation or contradiction of company policy, severe legal/security risk.\n"
        "- MEDIUM: Ambiguous language, partial mismatch, or missing required protective terms.\n"
        "- LOW: Fully compliant with company policy, zero or minimal risk.\n\n"
        "GROUNDING RULES (critical):\n"
        "- If the CONTRACT CLAUSE specifies a concrete number, timeframe, or frequency "
        "(e.g. a retention period, audit interval, notice period, or monetary cap), read the "
        "ENTIRE retrieved context for a sentence addressing that SAME specific parameter -- "
        "not just the first topically-related sentence. A single [Policy Result] block often "
        "contains multiple sentences on different sub-topics; don't stop at the first one that "
        "mentions the general subject.\n"
        "- Base your justification ONLY on the text given in RETRIEVED COMPANY POLICY CONTEXT below. "
        "Do not cite external laws, regulations, or 'typical'/'industry standard' figures (e.g. specific "
        "retention periods, liability amounts, or audit frequencies) unless that exact figure appears "
        "verbatim in the retrieved context.\n"
        "- If the retrieved context does not fully address the clause (e.g. it's silent on a specific "
        "number, timeframe, or requirement), say so explicitly in the justification instead of inferring "
        "or estimating a figure. Missing or incomplete policy coverage is itself valid grounds for a "
        "MEDIUM risk level -- state that as the reason rather than inventing a comparison point.\n"
        "- When you reference the policy, quote or closely paraphrase the specific retrieved sentence, "
        "and name which [Policy Result N] it came from.\n\n"
        "CRITICAL FORMATTING INSTRUCTION: You must respond ONLY with raw, valid JSON. "
        "Do NOT use markdown code blocks (e.g., do NOT write ```json or ```). "
        "Do NOT include any preamble, introductory text, or postscript. "
        "Your entire output must start with '{' and end with '}'.\n\n"
        "Return JSON with this structure:\n"
        "{\n"
        '  "risk_level": "HIGH | MEDIUM | LOW",\n'
        '  "justification": "Clear, objective breakdown of why this risk score was assigned, grounded '
        'only in the retrieved policy context and citing which Policy Result it came from.",\n'
        '  "recommendation": "Suggested modification or action for the legal team."\n'
        "}"
    )

    user_prompt = f"""
CONTRACT CLAUSE TITLE: {clause_title}
CONTRACT CLAUSE TEXT:
"{clause_text}"

RETRIEVED COMPANY POLICY CONTEXT:
{policy_context}
"""

    score_data = call_llm(provider, model_name, api_key, system_prompt, user_prompt)
    
    risk_level = score_data.get("risk_level", "MEDIUM").upper()
    justification = score_data.get("justification", "")
    recommendation = score_data.get("recommendation", "")

    if risk_level in ["HIGH", "MEDIUM"]:
        log_msg = log_for_human_review(
            contract_name=contract_name,
            clause_text=clause_text,
            risk_level=risk_level,
            justification=justification,
            session_id=session_id
        )
        print(f"MCP HITL Tool: {log_msg}")

    return {
        "clause_title": clause_title,
        "clause_text": clause_text,
        "risk_level": risk_level,
        "justification": justification,
        "recommendation": recommendation
    }

def run_risk_scoring_agent(retrieval_payload: dict, contract_name: str, provider: str, model_name: str, api_key: str, session_id: str = None) -> dict:
    """Runs the risk scoring agent across all retrieved clauses in a payload."""
    scored_clauses = []
    clauses = retrieval_payload.get("clauses", [])

    print(f" Risk Scorer evaluating {len(clauses)} clause(s)...")
    for clause in clauses:
        evaluated = score_clause_risk(contract_name, clause, provider, model_name, api_key, session_id=session_id)
        scored_clauses.append(evaluated)

    return {
        "contract_name": contract_name,
        "document_type": retrieval_payload.get("document_type", "Unknown"),
        "evaluations": scored_clauses
    }