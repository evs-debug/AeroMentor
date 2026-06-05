import pickle
import subprocess

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

with open("aviation_index.pkl", "rb") as file:
    database = pickle.load(file)

print("Knowledge base loaded!")

while True:

    question = input("\nAsk AeroMentor: ")

    if question.lower() == "exit":
        break

    question_embedding = model.encode([question])

    scores = []

    for document in database:

        similarity = cosine_similarity(
            question_embedding,
            [document["embedding"]]
        )[0][0]

        scores.append(
            (
                document["filename"],
                similarity,
                document["text"]
            )
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = scores[:3]

    context = ""

    for filename, score, text in top_docs:
        context += f"\n\n{text}"

    prompt = f"""
You are AeroMentor, an aviation instructor.

Use only the aviation information provided.

Context:
{context}

Question:
{question}
"""

    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True,
        text=True
    )

    print("\nTop Documents:")

    for filename, score, _ in top_docs:
        print(
            f"{filename}: {score:.4f}"
        )

    print("\nAeroMentor:")
    print(result.stdout)