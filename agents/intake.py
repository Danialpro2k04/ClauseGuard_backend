from agents.llm import call_llm

def run_intake_agent(contract_text: str, provider: str, model_name: str, api_key: str) -> dict:
    """Parses contract text and identifies risk-bearing clauses."""
    system_prompt = (
        "You are an expert Legal Intake Compliance Agent. Your job is to analyze contract text "
        "and identify key risk-bearing clauses (e.g., data security, liability, data retention, IP).\n\n"
        "GRANULARITY RULE: If a single paragraph contains multiple distinct obligations or "
        "restrictions (e.g., one sentence sets an audit frequency and a separate sentence "
        "prohibits a specific verification method like on-site inspection), extract EACH as "
        "its own separate entry in the clauses array -- even if they appear in the same "
        "paragraph of the source contract. Do not merge distinct restrictions into one "
        "clause_text just because they are textually adjacent.\n\n"
        "CRITICAL INSTRUCTION: Do NOT generate questions. Instead, formulate clear, direct, "
        "declarative STATEMENTS summarizing what the contract stipulates or permits. "
        "These statements will be semantically matched against company policy documents.\n\n"
        "CRITICAL FORMATTING INSTRUCTION: You must respond ONLY with raw, valid JSON. "
        "Do NOT use markdown code blocks (e.g., do NOT write ```json or ```). "
        "Do NOT include any preamble, introductory text, or postscript. "
        "Your entire output must start with '{' and end with '}'.\n\n"
        "Return JSON with this exact structure:\n"
        "{\n"
        '  "document_type": "NDA | MSA | Vendor Agreement | Unknown",\n'
        '  "clauses": [\n'
        '    {\n'
        '      "clause_title": "Title or summary of clause",\n'
        '      "clause_text": "Exact or verbatim snippet from the contract",\n'
        '      "compliance_statement": "Declarative statement of what the clause permits/requires (e.g., \'Data storage at rest is not required to be encrypted.\')"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = f"Analyze this contract text:\n\n{contract_text}"

    return call_llm(provider, model_name, api_key, system_prompt, user_prompt)