import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = {}

for filename in os.listdir("data"):
    if filename.endswith(".txt"):

        path = os.path.join("data", filename)

        with open(path, "r") as file:
            text = file.read()

        embedding = model.encode(text)

        documents[filename] = embedding

print("Embedded", len(documents), "documents")

for name in documents:
    print(name)