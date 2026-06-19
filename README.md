# AeroMentor

AeroMentor is an aviation-focused AI assistant built using Retrieval-Augmented Generation (RAG), semantic search, embeddings, and a local Llama 3 model.

The system retrieves relevant aviation knowledge from a custom aviation knowledge base and generates source-backed answers through a Streamlit web interface.

---

## Features

Semantic search using sentence embeddings
FAISS vector database for efficient retrieval
Retrieval-Augmented Generation (RAG)
Local Llama 3 inference through Ollama
Conversational Streamlit chat interface
Chat history and session management
Follow-up question support with context-aware retrieval
Aircraft-aware reranking system
Source-backed answers with citations

---

## Screenshots

### Home Screen

![Home Screen](docs/aeromentor-home.png)

### Example Question

![V1 Example](docs/aeromentor-v1.png)

### Boeing 787 Example

![B787 Example](docs/aeromentor-b787.png)
---

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
FAISS Vector Search
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
Streamlit Chat Interface

See the architecture diagram in:

`docs/architecture.png`

---

## Knowledge Base

Current knowledge base includes:

* Aircraft types
* Aerodynamics
* Flight operations
* Navigation systems
* Engine systems
* Airport operations
* Aviation procedures

**Total aviation knowledge chunks:** 167

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


PDF knowledge ingestion
Image understanding
Aviation diagram analysis
Expanded aviation knowledge base
Advanced evaluation benchmarks
Cloud deployment

---

## Project Structure

aviation-ai/
├── app.py
├── aviation_chunk_index.pkl
├── aviation_faiss.index
├── data/
├── docs/
├── src/
├── tests/
└── README.md

## Author

Eva Sharma

AI & Software Development Portfolio Project


