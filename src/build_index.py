import os
import pickle

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

database = []

for filename in os.listdir("data"):

    if filename.endswith(".txt"):

        path = os.path.join("data", filename)

        with open(path, "r") as file:
            text = file.read()

        embedding = model.encode(text)

        database.append(
            {
                "filename": filename,
                "text": text,
                "embedding": embedding
            }
        )

with open("aviation_index.pkl", "wb") as file:
    pickle.dump(database, file)

print(f"Indexed {len(database)} documents!")