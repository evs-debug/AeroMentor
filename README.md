# AeroMentor

AeroMentor is an aviation-focused AI assistant built using Retrieval-Augmented Generation (RAG), semantic search, FAISS vector search, and a local Llama 3 model.

The system retrieves aviation knowledge from a custom knowledge base, supports conversational question answering, and allows users to upload PDF documents and chat with them through a Streamlit web interface.

---

## Features

- Semantic search using sentence embeddings
- FAISS vector database for efficient retrieval
- Retrieval-Augmented Generation (RAG)
- Local Llama 3 inference through Ollama
- Conversational Streamlit chat interface
- Chat history and session management
- Follow-up question support with context-aware retrieval
- Aircraft-aware reranking system
- Aircraft specification cards
- Retrieval analytics dashboard
- Upload PDF and chat with user-provided documents
- Source-backed answers with citations

---

## Screenshots

### Home Screen
![Home Screen](screenshots/home1.png)

### Aircraft Specification Cards
![Aircraft Specs](screenshots/aircraft_specs.png)

### Retrieval Dashboard
![Retrieval Dashboard](screenshots/retrieval_metrics.png)

### Upload PDF
![PDF Upload](screenshots/pdf.png)

### Chat With Uploaded PDF
![PDF Question Answering](screenshots/pdf1.png)

## Architecture

AeroMentor follows a Retrieval-Augmented Generation (RAG) pipeline:

User Question
        ↓
Conversational Context
        ↓
Entity Tracking
        ↓
Embedding Model (all-MiniLM-L6-v2)
        ↓
Question Embedding
        ↓
FAISS Knowledge Base Search
              +
Uploaded PDF Retrieval
        ↓
Aircraft-Aware Reranking
        ↓
Top Relevant Chunks
        ↓
Context Assembly
        ↓
Llama 3 (Ollama)
        ↓
Answer Generation
        ↓
Source Citations
        ↓
Streamlit Interface

See the architecture diagram in:

`docs/architecture.png`

---

## Knowledge Base

Current knowledge base includes:

- Aircraft types
- Aerodynamics
- Flight operations
- Navigation systems
- Engine systems
- Airport operations
- Aviation procedures
- FAA Pilot's Handbook of Aeronautical Knowledge
- FAA Airplane Flying Handbook

**Total aviation knowledge chunks:** 6343+

---

## Technologies Used

* Python
* Ollama
* Llama 3
* Sentence Transformers
* Streamlit
* Scikit-Learn
* Git
* GitHub

---

## Installation

Clone the repository:

```bash
git clone https://github.com/evs-debug/AeroMentor.git
cd AeroMentor
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running AeroMentor

Launch the Streamlit application:

```bash
python -m streamlit run app.py
```

---

### Conversational Memory and Context-Aware Retrieval

AeroMentor supports follow-up questions by tracking aircraft entities mentioned in previous user queries. When a user asks a context-dependent question such as "What engines does it use?", the system automatically identifies the referenced aircraft and reformulates the retrieval query before performing semantic search.

Example:

User: What is the A350?
User: What engines does it use?

Retrieval Query:
A350 What engines does it use?

This enables more accurate retrieval of relevant knowledge chunks and creates a more natural conversational experience.


## Future Improvements

- Aviation Topic Explorer
- Learning Mode with quizzes and explanations
- Enhanced source citations
- Search analytics visualizations
- Multi-document upload and comparison
- Aviation diagram understanding
- OCR support for scanned PDFs
- Cloud deployment

---

## Dynamic PDF Question Answering

Users can upload aviation manuals, lecture notes, and PDF documents and ask questions directly from the uploaded content.

### Example

PDF:
Untitled document.pdf

Question:
What engines does the SkyDragon X1 use?

Answer:
The SkyDragon X1 aircraft uses Phoenix-900 engines.

Source:
Untitled document.pdf | Chunk 0

## Project Structure

aviation-ai/
├── app.py
├── data/
│   ├── aircraft_specs.json
│   ├── phak.txt
│   └── airplane_flying_handbook.txt
├── docs/
├── pdfs/
├── screenshots/
├── src/
│   ├── ingest_pdf.py
│   ├── build_chunk_index.py
│   └── build_faiss_index.py
├── tests/
├── aviation_chunk_index.pkl
├── aviation_faiss.index
└── README.md



## Upload PDF & Chat With It

Users can upload aviation manuals, lecture notes, or documents and ask questions directly from the uploaded content.

### Example

PDF: Untitled document.pdf

Question:
What engines does the SkyDragon X1 use?

Answer:
The SkyDragon X1 aircraft uses Phoenix-900 engines.

## Recent Updates

### Dynamic PDF Ingestion
- Upload arbitrary PDF documents
- Extract text using pypdf
- Generate embeddings on demand
- Perform semantic retrieval over uploaded documents
- Answer questions using retrieved document context


---

## Author

Eva Sharma
B.Tech Computer Science and Engineering
Vellore Institute of Technology (VIT)

AI • Retrieval-Augmented Generation • NLP • Software Engineering