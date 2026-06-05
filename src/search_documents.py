import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

question = input("Question: ")

question_embedding = model.encode([question])

scores = []

for filename in os.listdir("data"):

    if filename.endswith(".txt"):

        path = os.path.join("data", filename)

        with open(path, "r") as file:
            text = file.read()

        document_embedding = model.encode([text])

        similarity = cosine_similarity(
            question_embedding,
            document_embedding
        )[0][0]

        scores.append(
            (filename, similarity)
        )

scores.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nTop Matches:\n")

for filename, score in scores[:5]:
    print(
        f"{filename}: {score:.4f}"
    )