import os
import sys

# Ensure this script's directory is importable when run standalone (server/ is a sibling package)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.mcp_server import build_policy_collection

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")

COLLECTION_NAME = "company_policies"


def ingest_corpus():
    """Loads every file in CORPUS_DIR and ingests it into the shared
    'company_policies' Qdrant collection.

    This now delegates entirely to build_policy_collection() in
    server/mcp_server.py rather than re-implementing chunking, embedding,
    and upsert logic here. Previously this file and mcp_server.py each had
    their own copy of the RecursiveCharacterTextSplitter config; they drifted
    out of sync (500/50 in one place, changed to 1200/150 in only the other),
    which meant policy documents ingested via this script could be chunked
    differently than documents ingested via the API's build_policy_collection
    path -- silently changing retrieval behavior depending on which ingestion
    route was used. Having one function own the chunking config removes that
    entire class of bug.
    """
    print(f" Looking for corpus files in: {CORPUS_DIR}")
    if not os.path.exists(CORPUS_DIR):
        print(f" Error: Corpus directory does not exist at {CORPUS_DIR}")
        return

    policy_documents = []
    for file_name in os.listdir(CORPUS_DIR):
        file_path = os.path.join(CORPUS_DIR, file_name)

        # Ignore subdirectories or hidden files like .gitkeep
        if os.path.isfile(file_path) and not file_name.startswith('.'):
            print(f" Processing: {file_name}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f" Could not read {file_name}: {e}")
                continue
            policy_documents.append({"source": file_name, "text": content})

    if not policy_documents:
        print("No valid files found in corpus directory to ingest.")
        return

    print(f" Ingesting {len(policy_documents)} document(s) into '{COLLECTION_NAME}'...")
    build_policy_collection(collection_name=COLLECTION_NAME, policy_documents=policy_documents)
    print("Ingestion completed successfully!")


if __name__ == "__main__":
    ingest_corpus()
