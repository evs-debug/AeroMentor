import pickle
import subprocess
import re
import faiss
import numpy as np
import json
import time


import streamlit as st

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

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

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )



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
with open(
    "data/aircraft_specs.json",
    "r"
) as file:

    aircraft_specs = json.load(file)

for chunk in database:
    if chunk["filename"] == "a350.txt":
        print(
            f"Chunk {chunk['chunk_number']}"
        )
        print(chunk["text"])
        print("----------------") 

@st.cache_resource
def load_faiss_index():

    return faiss.read_index(
        "aviation_faiss.index"
    )

index = load_faiss_index()

uploaded_database = []

if uploaded_file:

    try:

        uploaded_file.seek(0)

        reader = PdfReader(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Could not read PDF: {e}"
        )

        st.stop()

    uploaded_text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            uploaded_text += (
                page_text + "\n"
            )
    
    st.write(
        f"Pages: {len(reader.pages)}"
    )

    st.write(
        f"Characters: {len(uploaded_text)}"
    )

    uploaded_chunks = []

    chunk_size = 1000

    for i in range(
        0,
        len(uploaded_text),
        chunk_size
    ):

        chunk = uploaded_text[
            i:i + chunk_size
        ].strip()

        if chunk:
            uploaded_chunks.append(
                chunk
            )

    embeddings = model.encode(
        uploaded_chunks,
        show_progress_bar=True
    ).astype("float32")

    for chunk_number, (chunk, embedding) in enumerate(
        zip(uploaded_chunks, embeddings)
    ):
        uploaded_database.append(
            {
                "filename": uploaded_file.name,
                "chunk_number": chunk_number,
                "text": chunk,
                "embedding": embedding
            }
        )

    st.write(
        f"Chunks: {len(uploaded_chunks)}"
    )

    st.write(
        f"Embeddings: "
        f"{len(uploaded_database)}"
    )

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
    
    start_time = time.time()

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

    selected_aircraft = None

    for plane in aircraft_specs:

        if plane in question_lower:
            selected_aircraft = plane
            break

    aircraft = [
        "a320",
        "a350",
        "a380",
        "b737",
        "b747",
        "b777",
        "b787"
    ]


    is_follow_up = any(
        word in question_lower.split()
        for word in follow_up_words
    )


    last_aircraft = ""

    for plane in aircraft:
        if plane in history_text.lower():
            last_aircraft = plane.upper()

    if is_follow_up and last_aircraft:

        search_query = (
            f"{last_aircraft} {question}"
        )

    else:
        search_query = question



    question_embedding = model.encode(
        [search_query]
    ).astype("float32")

    uploaded_scores = []

    for chunk in uploaded_database:

        embedding = np.array(
            chunk["embedding"],
            dtype="float32"
        )

        similarity = np.dot(
            question_embedding[0],
            embedding
        )

        uploaded_scores.append(
            (
                chunk["filename"],
                chunk["chunk_number"],
                similarity,
                chunk["text"]
            )
        )

    uploaded_scores.sort(
        key=lambda x: x[2],
        reverse=True
    )

    print("\nUPLOADED PDF SCORES:")

    for (
        filename,
        chunk_number,
        score,
        text
    ) in uploaded_scores:

        print(
            f"{filename}"
            f" | Chunk {chunk_number}"
            f" | Score {score}"
        )


   



    distances, indices = index.search(
        question_embedding,
        50
    )

    print("\nFAISS RESULTS:")

    for i in indices[0]:
        chunk = database[i]
        print(
            f"{chunk['filename']}"
            f" | Chunk {chunk['chunk_number']}"
        )

    scores = []

    for i in indices[0]:

        chunk = database[i]

        similarity = 0.0

        if "engine" in question_lower:
            if "engine" in chunk["text"].lower():
                similarity += 0.50

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

        position = list(
            indices[0]
        ).index(i)

        similarity = float(
            distances[0][position]
        )

    scores.sort(
        key=lambda x: x[2],
        reverse=True
    )

    top_chunks = scores[:8]

    all_scores = (
        scores +
        uploaded_scores
    )

    all_scores.sort(
        key=lambda x: x[2],
        reverse=True
    )

    top_chunks = all_scores[:8]

    source_files = set()

    for (
        filename,
        chunk_number,
        score,
        text
    ) in top_chunks:

        source_files.add(filename)

    print("\nTOP CHUNKS:")

    for filename, chunk_number, score, text in top_chunks:
        print(
            f"{filename}"
            f" | Chunk {chunk_number}"
        )
        print(text[:200])
        print("----------------")



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

    response_time = (
        time.time()
        - start_time
    )

    with st.chat_message("assistant"):
        st.write(clean_output)

    with st.expander(
        "📊 Retrieval Metrics"
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Knowledge Base",
                f"{len(database)} chunks"
            )

            st.metric(
                "Retrieved Chunks",
                len(top_chunks)
            )

        with col2:

            st.metric(
                "Sources Used",
                len(source_files)
            )

            st.metric(
                "Response Time",
                f"{response_time:.2f}s"
            )
        st.divider()

        st.write(
            "Top Retrieved Chunks"
        )

        for (
            filename,
            chunk_number,
            score,
            text
        ) in top_chunks:

            st.write(
                f"{filename} | "
                f"Chunk {chunk_number}"
            )


    if selected_aircraft:

        specs = aircraft_specs[
            selected_aircraft
        ]

        st.subheader(
            f"✈️ {specs['name']}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Passengers",
                specs["passengers"]
            )

            st.metric(
                "Range",
                specs["range"]
            )

        with col2:
            st.metric(
                "Cruise Speed",
                specs["cruise_speed"]
            )

            st.metric(
                "Engines",
                specs["engines"]
            )

        st.caption(
            specs["role"]
        )

    st.session_state.history.append(
        (
            question,
            clean_output
        )
    )

    source_names = {
        "phak.txt":
            "FAA Pilot's Handbook of Aeronautical Knowledge",

        "airplane_flying_handbook.txt":
            "FAA Airplane Flying Handbook"
    }

    sources = {}

    with st.expander("📚 Sources"):

        for (
            filename,
            chunk_number,
            score,
            text
        ) in top_chunks:

            if filename not in sources:
                sources[filename] = []

            sources[filename].append(
                chunk_number
            )

        for (
            filename,
            chunks
        ) in sources.items():

            display_name = source_names.get(
                filename,
                filename
            )

            st.write(
                f"📄 {display_name}"
            )

            st.caption(
                f"Retrieved Chunks: {len(chunks)}"
            )

            st.caption(
                "Chunk IDs: " +
                ", ".join(
                    str(chunk)
                    for chunk in chunks
                )
            )