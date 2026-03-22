import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Canonical project files and directories
METADATA_PATH = os.path.join(DATA_DIR, "hybrede_metadata_v5.json")
FILTERED_PAPERS_PATH = os.path.join(PROCESSED_DIR, "filtered_papers.json")
SCREENING_LOG_PATH = os.path.join(PROCESSED_DIR, "screening_log.json")
AUDIT_LOG_PATH = os.path.join(PROCESSED_DIR, "audit_log.json")
HARVESTED_PDFS_DIR = os.path.join(DATA_DIR, "harvested_pdfs")
FULLTEXT_DIR = os.path.join(DATA_DIR, "fulltext")
RAG_STORE_DIR = os.path.join(BASE_DIR, "rag_store")
RUN_MANIFEST_DIR = os.path.join(PROCESSED_DIR, "run_manifests")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_dir(path: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path
