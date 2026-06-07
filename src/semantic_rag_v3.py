import pickle
import subprocess

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

with open(
    "aviation_chunk_index.pkl",
    "rb"
) as file:

    database = pickle.load(file)

print("Chunk knowledge base loaded!")
while True:

    question = input("\nAsk AeroMentor: ")

    if question.lower() == "exit":
        break

    question_embedding = model.encode([question])

    scores = []

    for chunk in database:

        similarity = cosine_similarity(
            question_embedding,
            [chunk["embedding"]]
        )[0][0]

        scores.append(
            (
                chunk["filename"],
                chunk["chunk_number"],
                similarity,
                chunk["text"]
            )
        )
    scores.sort(
        key=lambda x: x[2],
        reverse=True
    )

    top_chunks = scores[:5]

    context = ""

    for filename, chunk_number, score, text in top_chunks:

        context += (
            f"\n\nSource: {filename}"
            f" | Chunk: {chunk_number}\n"
        )

        context += text
    prompt = f"""
You are AeroMentor, an expert aviation instructor.

Use ONLY the aviation context provided.

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

    print("\nTop Chunks:\n")

    for filename, chunk_number, score, text in top_chunks:

        print(
            f"{filename}"
            f" | Chunk {chunk_number}"
            f" | Score: {score:.4f}"
        )

    print("\nAeroMentor:")
    print(result.stdout)

    print("\nAeroMentor:")
    print(result.stdout)

    print("\nSources:")

    shown = set()

    for filename, chunk_number, score, text in top_chunks:

        source = (
            f"{filename}"
            f" | Chunk {chunk_number}"
        )

        if source not in shown:

            print(source)

            shown.add(source)