import pickle

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

with open(
    "tests/tests.txt",
    "r"
) as file:

    questions = file.readlines()

for question in questions:

    question = question.strip()

    if not question:
        continue

    question_embedding = model.encode([question])

    scores = []

    for chunk in database:

        similarity = cosine_similarity(
            question_embedding,
            [chunk["embedding"]]
        )[0][0]

        question_lower = question.lower()

        filename_lower = chunk["filename"].lower()

        if "a350" in question_lower and "a350" in filename_lower:
            similarity += 0.20

        if "a320" in question_lower and "a320" in filename_lower:
            similarity += 0.20

        if "a380" in question_lower and "a380" in filename_lower:
            similarity += 0.20

        if "b737" in question_lower and "b737" in filename_lower:
            similarity += 0.20

        if "b747" in question_lower and "b747" in filename_lower:
            similarity += 0.20

        if "b777" in question_lower and "b777" in filename_lower:
            similarity += 0.20

        if "b787" in question_lower and "b787" in filename_lower:
            similarity += 0.20 

        scores.append(
            (
                chunk["filename"],
                similarity
            )
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 50)
    print("QUESTION:")
    print(question)

    print("\nTOP 3 SOURCES:")

    for filename, score in scores[:3]:

        print(
            f"{filename}"
            f" ({score:.4f})"
        )