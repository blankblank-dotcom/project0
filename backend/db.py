"""
backend/db.py — The Local Titan: ChromaDB Vector Store
======================================================

Provides a portable, distribution-safe ChromaDB client for document
embeddings and semantic search.

Key design decisions:
  - SQLite version guard:  ChromaDB 1.0+ requires SQLite ≥ 3.35.0 for
    `RETURNING` clause support.  This check runs at import time so the
    error message is clear instead of a cryptic SQL syntax failure.
  - Portable persistence:  The Chroma data directory is always resolved
    relative to a user-writable AppData folder (frozen .exe) or the
    project root (dev mode).  Never uses an absolute dev-machine path.
  - Thread-safe singleton:  A module-level factory function returns the
    same `chromadb.ClientAPI` instance across all callers.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Optional, Any, Union
import base64

log = logging.getLogger("db")

# ═══════════════════════════════════════════════════════════════════════════
# SQLITE VERSION GUARD
# ═══════════════════════════════════════════════════════════════════════════
# ChromaDB 1.0+ uses SQL features (RETURNING, strict tables, generated
# columns) that require SQLite 3.35.0+.  Python's bundled sqlite3 on
# older Windows installs may ship 3.31 or 3.32.
#
# We check at import time so the failure is obvious and actionable.

_MINIMUM_SQLITE = (3, 35, 0)

if sqlite3.sqlite_version_info < _MINIMUM_SQLITE:
    _actual = sqlite3.sqlite_version
    _required = ".".join(str(v) for v in _MINIMUM_SQLITE)
    raise RuntimeError(
        f"Legacy SQLite detected: {_actual}\n"
        f"ChromaDB requires SQLite ≥ {_required}.\n\n"
        f"Fix options:\n"
        f"  1. Upgrade Python to 3.11+ (ships SQLite 3.39+)\n"
        f"  2. Install pysqlite3-binary:  pip install pysqlite3-binary\n"
        f"  3. Replace the sqlite3.dll in your Python install\n"
    )

log.info(f"SQLite version: {sqlite3.sqlite_version} (≥ {'.'.join(str(v) for v in _MINIMUM_SQLITE)} ✓)")


# ═══════════════════════════════════════════════════════════════════════════
# PORTABLE PERSISTENCE PATH
# ═══════════════════════════════════════════════════════════════════════════
def _get_chroma_persist_dir() -> Path:
    """Return a portable, writable directory for ChromaDB storage.

    ┌──────────────┬───────────────────────────────────────────────┐
    │ Context      │ Path                                          │
    ├──────────────┼───────────────────────────────────────────────┤
    │ Frozen .exe  │ %LOCALAPPDATA%/LocalTitan/chroma_db           │
    │ Dev (Win)    │ %LOCALAPPDATA%/LocalTitan/chroma_db           │
    │ Dev (other)  │ ./chroma_db  (project-relative)               │
    └──────────────┴───────────────────────────────────────────────┘

    The directory is created on first call if it doesn't exist.
    """
    if sys.platform == "win32":
        import tempfile
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        persist_dir = base / "LocalTitan" / "chroma_db"
    else:
        if getattr(sys, "frozen", False):
            persist_dir = Path.home() / ".local_titan" / "chroma_db"
        else:
            persist_dir = Path.cwd() / "chroma_db"

    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir


# ═══════════════════════════════════════════════════════════════════════════
# CHROMADB CLIENT (Thread-safe Singleton)
# ═══════════════════════════════════════════════════════════════════════════
_client: Optional[object] = None  # chromadb.ClientAPI
_client_lock = threading.Lock()


def get_chroma_client():
    """Return a persistent ChromaDB client (singleton).

    Uses a module-level lock to ensure thread safety during first
    initialization.  Subsequent calls return the cached instance.

    Returns:
        chromadb.ClientAPI — persistent client backed by SQLite.
    """
    global _client

    if _client is not None:
        return _client

    with _client_lock:
        # Double-check inside lock
        if _client is not None:
            return _client

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed.\n"
                "Install with: pip install chromadb"
            )

        persist_dir = _get_chroma_persist_dir()
        log.info(f"ChromaDB persist directory: {persist_dir}")

        # Modern ChromaDB >= 0.4.0 uses PersistentClient
        _client = chromadb.PersistentClient(path=str(persist_dir))

        log.info("ChromaDB client initialized ✓")
        return _client


# ═══════════════════════════════════════════════════════════════════════════
# ENCRYPTION MANAGER (Fernet-based)
# ═══════════════════════════════════════════════════════════════════════════
class SecurityManager:
    """Handles value-level encryption for database fields.
    
    Uses cryptography.fernet (AES-128 in CBC mode with HMAC-SHA256)
    to encrypt text blobs before storage.
    """
    def __init__(self, master_key: Optional[str] = None):
        self.fernet = None
        if master_key:
            try:
                from cryptography.fernet import Fernet
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                
                # Derive a 32-byte key from the master_key (password)
                salt = b"local_titan_salt_v1" # Constant salt for local persistence
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
                self.fernet = Fernet(key)
                log.info("Encryption layer active (PBKDF2 Derived Key) ✓")
            except Exception as e:
                log.error(f"Failed to initialize encryption: {e}")

    def encrypt(self, plain_text: str) -> str:
        """Encrypt string to base64-encoded ciphertext."""
        if not self.fernet or not plain_text:
            return plain_text
        return self.fernet.encrypt(plain_text.encode()).decode()

    def decrypt(self, cipher_text: str) -> str:
        """Decrypt base64-encoded ciphertext back to string."""
        if not self.fernet or not cipher_text:
            return cipher_text
        try:
            return self.fernet.decrypt(cipher_text.encode()).decode()
        except Exception:
            # If decryption fails (wrong key or plain text), return as-is
            return cipher_text

# Global security instance (updated by State)
_security = SecurityManager()

def set_master_password(password: str):
    global _security
    _security = SecurityManager(password)



def get_or_create_collection(name: str = "documents"):
    """Get or create a named ChromaDB collection.

    Args:
        name: Collection name (default: "documents").

    Returns:
        chromadb.Collection
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for embeddings
    )
    log.info(f"Collection '{name}': {collection.count()} document(s)")
    return collection


def index_document(
    content: str,
    metadata: dict,
    document_id: str,
    embedding: Optional[list[float]] = None
):
    """Add a document snippet to the vector store.
    
    Args:
        content: The text content to index.
        metadata: Source metadata (filename, page, timestamp).
        document_id: Unique ID for this entry.
        embedding: Pre-computed embedding vector (if available).
    """
    collection = get_or_create_collection()
    
    # Encrypt content and sensitive metadata before storage
    encrypted_content = _security.encrypt(content)
    encrypted_metadata = {
        k: (_security.encrypt(v) if isinstance(v, str) else v)
        for k, v in metadata.items()
    }
    
    try:
        kwargs = {
            "documents": [encrypted_content],
            "metadatas": [encrypted_metadata],
            "ids": [document_id]
        }
        if embedding:
            kwargs["embeddings"] = [embedding]
            
        collection.add(**kwargs)
        log.info(f"Indexed document: {document_id} ({len(content)} chars)")
    except Exception as e:
        log.error(f"Failed to index {document_id}: {e}")


def semantic_search(query_text: str, n_results: int = 5):
    """Search for relevant document snippets.
    
    Args:
        query_text: The search query.
        n_results: Max results to return.
        
    Returns:
        QueryResult object from ChromaDB.
    """
    collection = get_or_create_collection()
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        if results and "documents" in results:
            # Decrypt search results
            decrypted_docs = []
            for doc_list in results["documents"]:
                decrypted_docs.append([_security.decrypt(d) for d in doc_list])
            results["documents"] = decrypted_docs
            
            # Decrypt metadata
            if "metadatas" in results:
                decrypted_meta = []
                for meta_list in results["metadatas"]:
                    batch_meta = []
                    for m in meta_list:
                        batch_meta.append({
                            k: (_security.decrypt(v) if isinstance(v, str) else v)
                            for k, v in m.items()
                        })
                    decrypted_meta.append(batch_meta)
                results["metadatas"] = decrypted_meta
        return results
    except Exception as e:
        log.error(f"Search failed for '{query_text}': {e}")
        return None

