from rag.data_loader import load_data
from vector_db.faiss_index import *

data = load_data()

print("Preparing chunks...")
chunks = prepare_chunks(data)

print("Generating embeddings...")
embeddings = create_embeddings(chunks)

print("Building FAISS index...")
index = build_faiss_index(embeddings)

save_index(index)
save_chunks()

print("INDEX BUILT SUCCESSFULLY")
