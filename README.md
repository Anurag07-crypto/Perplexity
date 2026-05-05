# 🔍 Perplexitiy

**Advanced Research & Information Extraction Platform**

Perplexitiy is an AI-powered platform designed for deep research, intelligent information extraction, and document-based question answering. It combines multiple agents with a Retrieval-Augmented Generation (RAG) system for comprehensive knowledge management.

---

## 🌟 Features

### 🤖 Perplexity Agent
- **Fast Mode**: ⚡ Quick responses for immediate queries
- **Deep Search**: 🔎 In-depth research with comprehensive results
- **Mail Integration**: 📧 Email agent for automated email generation
- **Web Extraction**: 📄 Extract and process web page content

### 📚 RAG System
- **Document Upload**: Upload `.txt` files to build a knowledge base
- **Smart Indexing**: Automatic document processing and embedding generation
- **Vector Search**: Retrieve relevant documents using semantic similarity
- **LLM Integration**: Generate contextual answers using retrieved documents

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - Interactive UI
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) - REST API server
- **LLM**: [Groq](https://groq.com/) - llama-3.1-8b-instant model
- **Embeddings**: [SentenceTransformer](https://www.sbert.net/) - BAAI/bge-small-en-v1.5
- **Vector Store**: [ChromaDB](https://docs.trychroma.com/) - Persistent vector database
- **Text Processing**: [LangChain](https://python.langchain.com/) - Document handling & splitting
- **ML Libraries**: NumPy, SciPy, scikit-learn

---

## 📁 Project Structure

```
Perplexitiy_mini/
├── frontend_server/
│   └── frontend.py        # Streamlit UI application
|   └── services.py        
├── backend_server/
│   └── backend.py           # FastAPI backend server
├── RAG/
│   ├── data_ingestion.py    # Document loading & splitting
│   ├── embedding_manager.py # Embedding generation
│   ├── vector_store.py      # ChromaDB integration
│   └── retriever.py         # Document retrieval logic
├── perplexitiy/
│   ├── deep_research_agent.py
│   ├── email_agent.py
│   ├── webpages_extract_agent.py
│   └── main.py
├── fetched_data/
│   ├── text_docs/           # Uploaded documents
│   └── deep_Search_results/ # Search results storage
├── vector_store_data/       # ChromaDB persistent storage
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md               # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip or conda package manager
- Virtual environment (recommended)

### Step 1: Clone/Setup Repository
```bash
cd Perplexitiy_mini
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
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
GROQ_API_KEY=your_email_password_here
```

Get your API keys:
- **Groq API Key**: https://console.groq.com/
- **Tavily API Key**: https://tavily.com/

---

## 🚀 Running the Application

### Option 1: Run Frontend Only (Streamlit)
```bash
streamlit run frontend_server/frontend.py
```
The frontend will be available at `http://localhost:8501`

### Option 2: Run Full Stack (Backend + Frontend)

**Terminal 1 - Start Backend Server:**
```bash
cd backend_server
python backend.py
```
Backend will run on `http://127.0.0.1:8000`

**Terminal 2 - Start Frontend:**
```bash
streamlit run frontend_server/frontend.py
```
Frontend will run on `http://localhost:8501`

---

## 📖 Usage Guide

### Using the Perplexity Agent

1. **Navigate to "🤖 Perplexity Agent" tab**
2. **Enter your query** in the text area
3. **Select an agent mode**:
   - Fast: Quick answers
   - deep_search: Comprehensive research
   - mail_sender: Generate emails
   - extract_pages: Extract web content
4. **Click "🚀 Submit Query"** and wait for results

### Using the RAG System

#### Indexing Documents:
1. **Navigate to "📚 RAG System" tab**
2. **Upload a `.txt` file** in the "Document Upload & Indexing" section
3. **Click "📥 Index Document"**
4. Wait for confirmation: ✅ File indexed successfully

#### Searching Documents:
1. **Enter your question** in the "Search Knowledge Base" section
2. **Click "🔎 Search"**
3. **Review results** - The LLM will generate answers based on indexed documents

#### Example Queries:
- "What is the main topic of this document?"
- "Summarize the key points"
- "Find information about [specific topic]"

---

## 🔄 RAG Pipeline

```
Document Upload
    ↓
Save to Disk (fetched_data/text_docs/)
    ↓
Load & Ingest (DirectoryLoader)
    ↓
Split into Chunks (RecursiveCharacterTextSplitter)
    ↓
Generate Embeddings (SentenceTransformer)
    ↓
Store in Vector DB (ChromaDB)
    ↓
Query & Retrieve (Semantic Search)
    ↓
Generate Response (Groq LLM)
```

---

## 📊 Configuration

### Embedding Model
- **Model**: BAAI/bge-small-en-v1.5
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
- **Database**: ChromaDB (Persistent)
- **Collection**: "Text_Data"
- **Location**: `vector_store_data/`

---

## 🔍 Key Components

### Data Ingestion (`RAG/data_ingestion.py`)
- Loads documents from `fetched_data/text_docs/`
- Splits text into manageable chunks
- Preserves document metadata

### Embedding Manager (`RAG/embedding_manager.py`)
- Generates semantic embeddings using SentenceTransformer
- Batch processes multiple texts
- Error handling for embedding failures

### Vector Store (`RAG/vector_store.py`)
- Manages persistent ChromaDB storage
- Adds documents with metadata
- Handles embedding format conversion

### Retriever (`RAG/retriever.py`)
- Performs semantic search
- Filters by similarity threshold
- Returns ranked results with metadata

---

## 🐛 Troubleshooting

### Issue: "No documents indexed yet"
- **Solution**: Upload a `.txt` file and click "📥 Index Document" first
- Check that `fetched_data/text_docs/` directory has files

### Issue: "Cannot connect to backend server"
- **Solution**: Ensure FastAPI server is running on port 8000
- Run: `python backend_server/backend.py`

### Issue: "Embeddings not generated"
- **Solution**: Check internet connection for model download
- Verify SentenceTransformer is properly installed

### Issue: Empty search results
- **Solution**: Try a more specific query
- Ensure documents are properly indexed
- Check document content relevance

---

## 📝 Logging

Application logs are stored in `logs/app.log` with the following information:
- **Timestamp**: When the event occurred
- **Level**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Source**: Which module logged the message
- **Message**: Detailed description

---

## 🔐 Security Notes

- Keep your `.env` file private and out of version control
- Don't commit API keys to the repository
- Use environment variables for sensitive data
- Implement authentication for production deployments

---

## 🤝 API Endpoints

### Backend Server (`FastAPI`)
- **POST** `/server` - Process query with selected tool
  ```json
  {
    "query": "your query",
    "tools": "fast|deep_search|mail_sender|extract_pages"
  }
  ```

---

## 📦 Dependencies

See `requirements.txt` for the complete list. Key packages:
- langchain==0.1.0+
- streamlit==1.30+
- fastapi==0.100+
- chromadb==0.4+
- sentence-transformers==2.2+
- langchain-groq==0.1+

---

## 🚀 Future Enhancements

- [ ] Multi-document support with source tracking
- [ ] User authentication and session management
- [ ] Advanced filtering and metadata-based search
- [ ] Document versioning and history
- [ ] Export search results to various formats
- [ ] Performance optimization for large datasets
- [ ] Web UI improvements and dark mode
- [ ] Custom embedding model support

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 💬 Support

For issues, questions, or contributions:
1. Check existing documentation
2. Review logs in `logs/app.log`
3. Verify all dependencies are installed
4. Ensure `.env` file is properly configured

---

## 🎯 Quick Start Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with API keys
- [ ] Backend server running (optional)
- [ ] Frontend started with Streamlit
- [ ] Test with sample document upload

---

**Built with ❤️ for Advanced Research and AI-Powered Information Extraction**
