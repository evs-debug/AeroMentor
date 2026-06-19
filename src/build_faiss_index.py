import pickle
import faiss
import numpy as np

with open(
    "aviation_chunk_index.pkl",
    "rb"
) as file:

    database = pickle.load(file)

embeddings = []

for chunk in database:

    embeddings.append(
        chunk["embedding"]
    )

embeddings = np.array(
    embeddings,
    dtype="float32"
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

faiss.write_index(
    index,
    "aviation_faiss.index"
)

print(
    f"FAISS index built with "
    f"{index.ntotal} vectors."
)