import json
import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_chroma import Chroma  
from langchain.embeddings.base import Embeddings

# CONFIG
CHUNKS_PATH = "chunks.json"
CHECKPOINT_PATH = "embed_checkpoint.json"
CHROMA_DIR = "chroma_db_nomic"
BATCH_SIZE = 500

# Custom embedding wrapper for LangChain
class NomicEmbedder(Embeddings):
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

# Load documents from JSON
def load_documents(path=CHUNKS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    documents = []
    for i, entry in enumerate(raw_chunks):
        text = entry.get("page_content", "")
        metadata = entry.get("metadata", {})
        metadata["chunk_id"] = metadata.get("chunk_id", f"chunk_{i}")
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

# Load checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

# Save checkpoint
def save_checkpoint(done_ids):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(done_ids), f)

# Main embedding loop
def embed_and_persist():
    all_docs = load_documents()
    done_ids = load_checkpoint()

    docs_to_embed = [doc for doc in all_docs if doc.metadata["chunk_id"] not in done_ids]
    if not docs_to_embed:
        print("All documents already embedded.")
        return

    embedder = NomicEmbedder()
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)
    db._collection  # Force init

    print(f"Starting embedding of {len(docs_to_embed)} new documents...")

    for i in tqdm(range(0, len(docs_to_embed), BATCH_SIZE), desc="Embedding batches"):
        batch = docs_to_embed[i:i + BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        embeddings = embedder.embed_documents(texts)

        db.add_documents(batch, embeddings=embeddings)

        for doc in batch:
            done_ids.add(doc.metadata["chunk_id"])
        save_checkpoint(done_ids)

        time.sleep(0.1)

    print(f"Embedded {len(done_ids)} total documents. Saved to {CHROMA_DIR}")

if __name__ == "__main__":
    embed_and_persist()