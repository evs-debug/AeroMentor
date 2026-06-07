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

    if (
        "Definition:" in text
        and "Key Facts:" in text
        and "Importance:" in text
    ):

        sections = text.split("\n\n")

        for section in sections:

            section = section.strip()

            if section:
                chunks.append(section)

    else:

        sections = text.split("\n\n")

        for section in sections:

            section = section.strip()

            if section:
                chunks.append(section)

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