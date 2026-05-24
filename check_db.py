# Add this to check_db.py and run it
from main import VECTOR_STORE_PATH
from rag_pipeline import VectorStore

vs = VectorStore(persist_directory=VECTOR_STORE_PATH)
all_docs = vs.collection.get()

from collections import Counter
sources = Counter(m['source_file'] for m in all_docs['metadatas'])
print("Chunks per PDF:")
for source, count in sources.items():
    print(f"  {source}: {count} chunks")