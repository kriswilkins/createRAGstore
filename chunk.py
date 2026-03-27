import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

with open("docs.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

docs_list = [Document(**doc) for doc in raw_docs]
print(f"Loaded {len(docs_list)} documents")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0
)

doc_splits = []
for doc in tqdm(docs_list, desc="Chunking documents", unit="doc"):
    chunks = text_splitter.split_documents([doc])
    doc_splits.extend(chunks)

print(f"[✅] Total chunks created: {len(doc_splits)}")

temp_path = "chunks.json.tmp"
final_path = "chunks.json"

with open(temp_path, "w", encoding="utf-8") as f:
    json.dump([chunk.dict() for chunk in doc_splits], f, ensure_ascii=False, indent=2)

# Atomic rename
import os
os.replace(temp_path, final_path)

print(f"Saved chunks to {final_path}")