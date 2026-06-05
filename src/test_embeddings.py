from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

question = "What helps reduce fuel burn?"

chunk1 = "Winglets improve fuel efficiency."
chunk2 = "The Boeing 747 entered service in 1970."

question_embedding = model.encode([question])

chunk1_embedding = model.encode([chunk1])
chunk2_embedding = model.encode([chunk2])

similarity1 = cosine_similarity(
    question_embedding,
    chunk1_embedding
)

similarity2 = cosine_similarity(
    question_embedding,
    chunk2_embedding
)

print("Winglet similarity:", similarity1)
print("747 similarity:", similarity2)