# Perplexitiy

**Advanced Research, RAG, and Automation Platform**

Perplexitiy is an AI-powered application for deep research, web information extraction, document-based question answering, and recruiter outreach automation. It combines a Streamlit frontend, FastAPI backend, LangChain/LangGraph agents, Groq LLMs, and a ChromaDB-backed RAG pipeline.

---

## Features

### Perplexity Agent
- **Fast Mode**: Quick LLM responses for general queries.
- **Deep Search**: In-depth web research using Tavily-powered search.
- **Mail Integration**: Generate professional email content.
- **Web Extraction**: Extract and process web page content.

### RAG System
- **Document Upload**: Upload `.txt` files to build a local knowledge base.
- **Smart Indexing**: Split documents into chunks and generate embeddings.
- **Vector Search**: Retrieve relevant chunks using semantic similarity.
- **LLM Answers**: Generate contextual answers from retrieved documents.

### Recruiter Agent
- **CSV Upload**: Upload recruiter data from a `.csv` file.
- **Required Columns**: `name`, `email`, and `job_description`.
- **Personalized Emails**: Generate recruiter-specific emails tailored to each job description.
- **Batch Sending**: Send either the first batch or the full CSV list.
- **SMTP Delivery**: Sends emails through Gmail SMTP using an app password.

---

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Agent Framework**: LangChain and LangGraph
- **LLM**: Groq `llama-3.1-8b-instant`
- **Embeddings**: SentenceTransformer `BAAI/bge-small-en-v1.5`
- **Vector Store**: ChromaDB persistent storage
- **Text Processing**: LangChain document loaders and splitters
- **Data Handling**: pandas
- **Email Delivery**: Gmail SMTP

---

## Project Structure

```text
Perplexitiy_mini/
+-- frontend_server/
|   +-- frontend.py          # Streamlit UI with Perplexity, RAG, and Recruiter tabs
|   +-- services.py          # RAG and LLM service layer
|   +-- database_pdf.py
+-- backend_server/
|   +-- backend.py           # FastAPI backend server
+-- RAG/
|   +-- data_ingestion.py    # Document loading and splitting
|   +-- embedding_manager.py # Embedding generation
|   +-- vector_store.py      # ChromaDB integration
|   +-- retriever.py         # Document retrieval logic
+-- perplexitiy/
|   +-- deep_research_agent.py
|   +-- email_agent.py
|   +-- recruiter_agent.py   # CSV-based recruiter outreach automation
|   +-- webpages_extract_agent.py
|   +-- main.py              # Tool-based Perplexity agent orchestration
+-- fetched_data/
|   +-- text_docs/           # Uploaded text documents
|   +-- deep_Search_results/ # Deep search result output
+-- vector_store_data/       # ChromaDB persistent storage
+-- logs/                    # Application logs
+-- requirements.txt
+-- pyproject.toml
+-- .env                     # Local environment variables
+-- README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- pip, uv, or another Python package manager
- Virtual environment recommended
- Gmail app password if you want to use email sending

### Step 1: Open the Project

```bash
cd Perplexitiy_mini/Perplexity
```

### Step 2: Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
APP_PASSWORD=your_gmail_app_password_here
```

API and credential notes:
- **Groq API Key**: https://console.groq.com/
- **Tavily API Key**: https://tavily.com/
- **APP_PASSWORD**: Gmail app password used by the recruiter/email sending flow.

---

## Running the Application

### Option 1: Run Frontend Only

```bash
streamlit run frontend_server/frontend.py
```

The frontend will be available at `http://localhost:8501`.

### Option 2: Run Backend and Frontend

Terminal 1:

```bash
cd backend_server
python backend.py
```

Backend will run on `http://127.0.0.1:8000`.

Terminal 2:

```bash
streamlit run frontend_server/frontend.py
```

Frontend will run on `http://localhost:8501`.

---

## Usage Guide

### Perplexity Agent

1. Open the **Perplexity Agent** tab.
2. Enter your query.
3. Select a mode:
   - `Fast`: Quick LLM answer.
   - `deep_search`: Comprehensive research.
   - `mail_sender`: Email generation.
   - `extract_pages`: Web page extraction.
4. Click **Submit Query** and wait for the response.

### RAG System

Indexing documents:

1. Open the **RAG System** tab.
2. Upload a `.txt` file.
3. Click **Index Document**.
4. Wait for confirmation that the file was indexed.

Searching documents:

1. Enter your question in the search box.
2. Click **Search**.
3. Review the generated answer based on your indexed documents.

Example queries:
- "What is the main topic of this document?"
- "Summarize the key points."
- "Find information about a specific topic."

### Recruiter Agent

1. Open the **Recruiter Agent** tab.
2. Upload a `.csv` file with these columns:
   - `name`
   - `email`
   - `job_description`
3. Review the data preview and column validation.
4. Choose a batch size.
5. Enable **Send all emails** if you want to process the full CSV.
6. Click **Send Emails**.

The recruiter agent generates a personalized email for each row and sends it to the recruiter email address using Gmail SMTP.

Example CSV:

```csv
name,email,job_description
Jane Doe,jane@example.com,"Hiring for a Python ML Engineer with TensorFlow experience"
John Smith,john@example.com,"Looking for an AI Engineer skilled in PyTorch and data pipelines"
```

---

## RAG Pipeline

```text
Document Upload
    |
    v
Save to fetched_data/text_docs/
    |
    v
Load and ingest documents
    |
    v
Split into chunks
    |
    v
Generate embeddings
    |
    v
Store in ChromaDB
    |
    v
Query and retrieve relevant chunks
    |
    v
Generate response with Groq LLM
```

---

## Configuration

### Embedding Model
- **Model**: `BAAI/bge-small-en-v1.5`
- **Dimension**: 384
- **Location**: `RAG/embedding_manager.py`

### Text Chunking
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Location**: `RAG/data_ingestion.py`

### Retrieval Settings
- **Top-K Results**: 5 documents
- **Similarity Threshold**: 0.3
- **Location**: `RAG/retriever.py`

### Vector Store
- **Database**: ChromaDB persistent storage
- **Collection**: `Text_Data`
- **Location**: `vector_store_data/`

### Recruiter Email Settings
- **Sender**: configured in `perplexitiy/recruiter_agent.py`
- **SMTP Host**: `smtp.gmail.com`
- **SMTP Port**: `587`
- **Required Env Vars**: `GROQ_API_KEY`, `APP_PASSWORD`

---

## Key Components

### `perplexitiy/main.py`
- Routes Perplexity Agent requests to the selected tool.
- Supports `deep_search`, `mail_sender`, and `extract_pages` as backend tools.

### `perplexitiy/recruiter_agent.py`
- Reads recruiter records from a pandas DataFrame.
- Builds personalized job outreach prompts.
- Generates subject/body content with Groq.
- Sends each email using Gmail SMTP.

### `frontend_server/frontend.py`
- Provides three Streamlit tabs:
  - Perplexity Agent
  - RAG System
  - Recruiter Agent

### `frontend_server/services.py`
- Separates RAG retrieval and LLM answer generation from the UI layer.

### `RAG/`
- Handles ingestion, chunking, embeddings, vector storage, and retrieval.

---

## API Endpoints

### Backend Server

`POST /server`

```json
{
  "query": "your query",
  "tools": "Fast|deep_search|mail_sender|extract_pages"
}
```

Response:

```json
{
  "response": "agent response"
}
```

Note: the Recruiter Agent currently runs through the Streamlit frontend tab, not through the `/server` API endpoint.

---

## Troubleshooting

### No documents indexed yet
- Upload a `.txt` file from the RAG tab.
- Click **Index Document** before searching.
- Confirm files exist in `fetched_data/text_docs/`.

### Cannot connect to backend server
- Start the FastAPI server with `python backend_server/backend.py`.
- Confirm it is running at `http://127.0.0.1:8000`.

### Embeddings not generated
- Check your internet connection for first-time model download.
- Verify `sentence-transformers` is installed.

### Recruiter CSV has missing columns
- Make sure the CSV includes `name`, `email`, and `job_description`.
- Check for spelling differences or extra spaces in column names.

### Recruiter emails fail to send
- Confirm `APP_PASSWORD` is set in `.env`.
- Use a Gmail app password, not your normal Gmail password.
- Confirm `GROQ_API_KEY` is set.
- Check `logs/app.log` for SMTP or LLM errors.

---

## Logging

Application logs are stored in `logs/app.log`.

Logs include:
- Timestamp
- Log level
- Module name
- Request, indexing, retrieval, email, and error details

---

## Security Notes

- Keep `.env` private and out of version control.
- Do not commit API keys, Gmail credentials, or app passwords.
- Use app passwords for Gmail SMTP.
- Review recruiter CSV files before sending emails.
- Add authentication before deploying this application publicly.

---

## Dependencies

See `requirements.txt` for the complete dependency list. Key packages include:
- streamlit
- fastapi
- langchain
- langgraph
- langchain-groq
- chromadb
- sentence-transformers
- pandas
- python-dotenv

---

## Future Enhancements

- [ ] Add API endpoint support for recruiter batch processing.
- [ ] Add email draft preview before sending.
- [ ] Add per-recipient send status in the UI.
- [ ] Support attachments and resumes for recruiter emails.
- [ ] Add user authentication and session management.
- [ ] Add metadata filtering for RAG search.
- [ ] Export RAG and search results.
- [ ] Improve UI styling and dark mode support.

---

## Quick Start Checklist

- [ ] Python 3.10+ installed.
- [ ] Virtual environment created and activated.
- [ ] Dependencies installed with `pip install -r requirements.txt`.
- [ ] `.env` configured with `GROQ_API_KEY`.
- [ ] `.env` configured with `TAVILY_API_KEY` for deep search.
- [ ] `.env` configured with `APP_PASSWORD` for email sending.
- [ ] Backend server running if using Perplexity Agent API tools.
- [ ] Frontend started with Streamlit.
- [ ] Test RAG with a sample `.txt` file.
- [ ] Test Recruiter Agent with a small CSV batch first.

---

## License

This project is provided as-is for educational and research purposes.
