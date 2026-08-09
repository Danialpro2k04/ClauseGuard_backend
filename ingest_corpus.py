import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.mcp_server import build_policy_collection


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")

COLLECTION_NAME = "company_policies"


def ingest_corpus():
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
