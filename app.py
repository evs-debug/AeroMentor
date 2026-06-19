import pickle
import subprocess
import re

import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AeroMentor",
    page_icon="✈️"
)

st.title("✈️ AeroMentor")
st.write("Your Aviation AI Instructor")

with st.sidebar:

    st.header("⚙️ Settings")

    if st.button("🗑️ Clear Chat"):

        st.session_state.history = []

        st.rerun()


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_database():

    with open(
        "aviation_chunk_index.pkl",
        "rb"
    ) as file:

        return pickle.load(file)

model = load_model()
database = load_database()

if "history" not in st.session_state:

    st.session_state.history = []

question = st.chat_input(
    "Ask an aviation question:"
)

for user_question, assistant_answer in st.session_state.history:

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        st.write(assistant_answer)


if question:

    with st.chat_message("user"):
        st.write(question)

    history_text = ""

    for (
        user_question,
        assistant_answer
    ) in st.session_state.history[-3:]:

        history_text += (
            f"\nUser: {user_question}\n"
            f"Assistant: {assistant_answer}\n"
        )

    follow_up_words = [
        "it",
        "its",
        "they",
        "them",
        "that",
        "those",
        "these"
    ]

    question_lower = question.lower()


    is_follow_up = any(
        word in question_lower.split()
        for word in follow_up_words
    )

    if is_follow_up:
        search_query = (
            history_text +
            "\nCurrent Question: " +
            question
        )
    else:
        search_query = question

    question_embedding = model.encode(
        [search_query]
    )

    scores = []

    aircraft = [
        "a320",
        "a350",
        "a380",
        "b737",
        "b747",
        "b777",
        "b787"
    ]

    for chunk in database:

        similarity = cosine_similarity(
            question_embedding,
            [chunk["embedding"]]
        )[0][0]

        question_lower = question.lower()
        history_lower = history_text.lower()
        filename_lower = chunk["filename"].lower()

        for plane in aircraft:

            if (
                plane in search_query.lower()
                and plane in filename_lower
            ):
                similarity += 0.20



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
You are AeroMentor, an aviation instructor.

Answer ONLY using the provided context.

If the answer is present in the context,
answer using only that information.

If the answer truly cannot be found,
say:
"I do not have enough information in my knowledge base."

Previous Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""



    with st.spinner(
        "AeroMentor is thinking..."
    ):

        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True
        )



    clean_output = re.sub(
        r"\x1b\[[0-9;]*[A-Za-z]",
        "",
        result.stdout
    )

    with st.chat_message("assistant"):
        st.write(clean_output)

    st.session_state.history.append(
        (
            question,
            clean_output
        )
    )

    with st.expander("📚 Sources"):

        shown_sources = set()

        for (
            filename,
            chunk_number,
            score,
            text
        ) in top_chunks:

            source = (
                f"{filename}"
                f" | Chunk {chunk_number}"
            )

            if (
                score >= 0.60
                and source not in shown_sources
            ):

                shown_sources.add(source)

                st.write(source)