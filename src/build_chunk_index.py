import os
import pickle

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

database = []

for filename in os.listdir("data"):

    if not filename.endswith(".txt"):
        continue

    path = os.path.join("data", filename)

    with open(path, "r") as file:
        text = file.read()


    chunks = []

    chunk_size = 500

    for i in range(
        0,
        len(text),
        chunk_size
    ):
        chunk = text[
            i:i + chunk_size
        ].strip()

        if chunk:
            chunks.append(chunk)

    for chunk_number, chunk in enumerate(chunks):

        embedding = model.encode(chunk)

        database.append(
            {
                "filename": filename,
                "chunk_number": chunk_number,
                "text": chunk,
                "embedding": embedding
            }
        )

with open(
    "aviation_chunk_index.pkl",
    "wb"
) as file:

    pickle.dump(database, file)

print(
    f"Indexed {len(database)} chunks!"
)

with open(
    path,
    "r",
    encoding="utf-8"
) as file:
    text = file.read()