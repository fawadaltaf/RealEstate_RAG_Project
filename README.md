#   Real Estate & Builders — Company Knowledge Assistant (Module 08 RAG Project)

A reliable, source-grounded Retrieval-Augmented Generation (RAG) assistant built for **  Real Estate & Builders (BQRE)** using Python, ChromaDB, SentenceTransformers (`all-MiniLM-L6-v2`), Streamlit, and Groq/Gemini LLM APIs.

## Project Structure
- `app.py` — Streamlit web application with interactive Q&A, policy version filters, retrieved chunk inspector, 20-question evaluation runner, and **CSV/Excel export buttons**.
- `rag_engine.py` — Modular RAG backend handling metadata parsing (`version` & `status`), overlapping chunking, ChromaDB vector indexing, policy filtering, hybrid semantic + keyword scoring, prompt-injection defense, and grounded generation.
- `run_retrieval_experiments.py` — Automated script comparing `k=1, 3, 5` and chunk size/overlap settings (`300/30` vs `600/100`).
- `knowledge_base/` — 6 fictional real estate documents (~1,820+ words) covering company overview, 9 active properties, current policy (`v2.0`), archived policy (`v1.0`), operations/FAQs, and a prompt-injection test note.
- `requirements.txt` — Python dependencies.
- `.env.example` — Environment variable template for Groq or Gemini API keys.

## Installation Steps
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 2. Install required packages
pip install -r requirements.txt

# 3. Configure your free API key (Groq or Gemini)
copy .env.example .env        # Windows (or 'cp .env.example .env' on Mac/Linux)
# Open .env and paste your GROQ_API_KEY or GEMINI_API_KEY
```

## Command to Run the Application
```bash
streamlit run app.py
```

## Demonstration Video Link
- **3–5 Minute Demo Video:** = https://drive.google.com/file/d/1lCPm947Ozhv4Kq33UNgbhAjs_gje20mF/view?usp=sharing

