import os
import subprocess

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

while True:

    question = input("\nAsk AeroMentor: ")

    if question.lower() == "exit":
        break

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
                (filename, similarity, text)
            )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = scores[:3]

    context = ""

    for filename, score, text in top_docs:

        context += f"\n\nDocument: {filename}\n"
        context += text

    prompt = f"""
You are AeroMentor, an expert aviation instructor.

Use the provided aviation documents to answer.

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