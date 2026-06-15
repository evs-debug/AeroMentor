AeroMentor

AeroMentor is an aviation-focused AI assistant built using Retrieval-Augmented Generation (RAG), semantic search, embeddings, and a local Llama 3 model.

The system retrieves relevant aviation knowledge from a custom aviation knowledge base and generates source-backed answers through a Streamlit web interface.

Features
Semantic search using sentence embeddings
Chunk-based retrieval system
Retrieval-Augmented Generation (RAG)
Local Llama 3 inference through Ollama
Streamlit web application
Source citations for transparency
Benchmarking and retrieval evaluation
Aircraft-aware retrieval optimization
Architecture

AeroMentor follows a Retrieval-Augmented Generation pipeline:

User Question
↓
Embedding Model (all-MiniLM-L6-v2)
↓
Question Embedding
↓
Cosine Similarity Search
↓
Aviation Knowledge Base (167 Chunks)
↓
Top 5 Relevant Chunks
↓
Context Assembly
↓
Llama 3 (Ollama)
↓
Answer Generation
↓
Source Citations
↓
Streamlit UI

Knowledge Base

Current knowledge base contains:

Aircraft types
Aerodynamics
Navigation systems
Flight operations
Engine systems
Airport operations
Aviation procedures

Total knowledge chunks: 167

Technologies Used
Python
Ollama
Llama 3
Sentence Transformers
Streamlit
Scikit-Learn
Git
GitHub
Installation

Clone the repository:

git clone https://github.com/evs-debug/AeroMentor.git
cd AeroMentor

Create a virtual environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Running AeroMentor

Start the Streamlit application:

python -m streamlit run app.py
Future Improvements
Conversation memory
Chat history
Image understanding
Aviation diagram analysis
FAISS vector database
Expanded aviation knowledge base
Author

Eva Sharma

AI & Software Development Portfolio Project