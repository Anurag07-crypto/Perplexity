from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os
import streamlit as st
import requests
from datetime import datetime

from RAG.data_ingestion import data_ingest, splitter
from RAG.embedding_manager import EmbeddingManager
from RAG.vector_store import VectorStore
from frontend_server.services import RAGService, LLMService
from perplexitiy.recruiter_agent import processing
import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

load_env = load_dotenv()
os.getenv("GROQ_API_KEY")

# Initialize session state for vector store management
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = EmbeddingManager()
if "data_initialized" not in st.session_state:
    st.session_state.data_initialized = False
if "rag_service" not in st.session_state:
    st.session_state.rag_service = RAGService(
        embedding_manager=st.session_state.embedding_manager,
        vector_store=st.session_state.vector_store
    )
if "llm_service" not in st.session_state:
    st.session_state.llm_service = LLMService()

vector_store = st.session_state.vector_store
embedding_manager = st.session_state.embedding_manager
rag_service = st.session_state.rag_service
llm_service = st.session_state.llm_service

def initialize_data(force_reload=False):
    """Initialize vector store with documents from text_docs directory"""
    
    
    if st.session_state.data_initialized and not force_reload:
        logger.info("DATA already LOADED")
        return
    
    try:
        data = data_ingest()
        if not data:
            logger.warning("No documents found in text_docs directory")
            return
        
        chunks = splitter(data)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
        logger.info(f"DATA LOADED successfully: {len(chunks)} chunks indexed")
        st.session_state.data_initialized = True
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")
        raise




# Backend API configuration
BACKEND_URL = "http://127.0.0.1:8000"

# Page configuration
st.set_page_config(
    page_title="Perplexitiy",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1em;
        margin-bottom: 2em;
    }
    
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        font-size: 1.1em;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    .response-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
    }
    
    .tool-card {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔍 Perplexitiy</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Research & Information Extraction Platform</div>', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("📋 About")
    st.info("Perplexitiy is an AI-powered platform for deep research, data extraction, and intelligent information retrieval.")
    st.divider()
    st.header("⚙️ Settings")
    backend_status = st.radio("Backend Status", ["Auto-detect", "Online", "Offline"])
    request_timeout = st.slider("Request Timeout (seconds)", 30, 600, 500, 30)

# Main tabs
tab_1, tab_2, tab_3 = st.tabs(["🤖 Perplexity Agent", "📚 RAG System", "📧 Recruiter Agent"])

# ==================== TAB 1: PERPLEXITY AGENT ====================
with tab_1:
    st.header("🤖 Perplexity Agent")
    
    col_input, col_config = st.columns([2, 1], gap="large")
    
    with col_input:
        st.subheader("Query Configuration")
        query = st.text_area(
            "Enter your query",
            placeholder="Ask anything you want to research...",
            height=100,
            label_visibility="collapsed"
        )
        
    with col_config:
        st.subheader("Agent Selection")
        tools = st.selectbox(
            "Select Agent",
            ["Fast", "deep_search", "mail_sender", "extract_pages"],
            label_visibility="collapsed"
        )
        
        tool_descriptions = {
            "Fast": "⚡ Quick responses",
            "deep_search": "🔎 Deep research mode",
            "mail_sender": "📧 Email integration",
            "extract_pages": "📄 Web page extraction"
        }
        st.caption(tool_descriptions.get(tools, ""))
    
    # Submit button
    submit_col = st.columns([1, 3, 1])[1]
    with submit_col:
        submit_button = st.button("🚀 Submit Query", use_container_width=True)
    
    if submit_button:
        if not query.strip():
            st.error("❌ Please enter a query before submitting.")
        else:
            try:
                # Show processing indicators
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                response_placeholder = st.empty()
                
                with progress_placeholder.container():
                    progress_bar = st.progress(0, text="Initializing...")
                
                # Update progress
                progress_bar.progress(25, text="Sending request to server...")
                
                # Make API request
                res = requests.post(
                    f"{BACKEND_URL}/server",
                    json={"query": query, "tools": tools},
                    timeout=request_timeout
                )
                
                progress_bar.progress(75, text="Processing response...")
                
                if res.status_code == 200:
                    data = res.json()
                    response = data.get("response", "⚠️ No response from server")
                    
                    progress_bar.progress(100, text="✅ Complete!")
                    
                    # Display response in enhanced format
                    with response_placeholder.container():
                        st.markdown("---")
                        st.success("✅ Response received successfully!")
                        st.markdown('<div class="response-box">', unsafe_allow_html=True)
                        st.markdown(response)
                        if tools == "deep_search":
                            dir_path = Path(__file__).parent.parent / "fetched_data" / "deep_Search_results"/f"deep_search.txt"
                            
                            if dir_path.exists():
                                with open(dir_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                    st.markdown(content)
                                    
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Footer with timestamp
                        st.caption(f"📅 Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    progress_bar.progress(0)
                    st.error(f"❌ Server Error ({res.status_code}): {res.text}")

            except requests.exceptions.Timeout:
                st.error(f"⏱️ Request timeout after {request_timeout} seconds. Please try with a simpler query or increase timeout.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to backend server. Please ensure it's running on http://127.0.0.1:8000")
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

# ==================== TAB 2: RAG SYSTEM ====================
with tab_2:
    st.header("📚 RAG System")
    
    col_1, col_2 = st.columns(2, gap="large")
    
    with col_1:
        st.subheader("📤 Document Upload & Indexing")
        st.info("Upload your documents to build a knowledge base for intelligent retrieval and question answering.")
        
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["txt"],
            label_visibility="collapsed"
        )
        path = Path(__file__).parent.parent / "fetched_data" / "text_docs"
        path.mkdir(parents=True, exist_ok=True)
        if uploaded_file:
            actual_path = path / uploaded_file.name
            st.success(f"✅ File selected: {uploaded_file.name}")
            if st.button("📥 Index Document"):
                with st.spinner("Indexing document..."):
                    try:
                        # Step 1: Save file to disk
                        with open(actual_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        logger.info(f"File saved: {uploaded_file.name}")
                        
                        # Step 2: Ingest the document from disk
                        data = data_ingest()
                        if not data:
                            st.error(f"❌ No documents found in directory")
                            logger.error("No documents found to ingest")
                            raise ValueError("No documents found")
                        
                        # Step 3: Split documents into chunks
                        chunks = splitter(data)
                        logger.info(f"Document split into {len(chunks)} chunks")
                        
                        # Step 4: Generate embeddings
                        texts = [doc.page_content for doc in chunks]
                        embeddings = embedding_manager.generate_embeddings(texts)
                        logger.info(f"Generated {len(embeddings)} embeddings")
                        
                        # Step 5: Add to vector store
                        vector_store.add_documents(chunks, embeddings)
                        logger.info(f"Added {len(chunks)} documents to vector store")
                        
                        # Step 6: Mark data as initialized for retrieval
                        st.session_state.data_initialized = True
                        
                        st.success(f"✅ File indexed successfully: {uploaded_file.name}")
                        st.success(f"✅ Added {len(chunks)} chunks to vector store")
                        logger.info(f"Document fully indexed: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"❌ Error indexing document: {str(e)}")
                        logger.error(f"Error indexing document: {str(e)}")
                        import traceback
                        traceback.print_exc()
    
    with col_2:
        st.subheader("🔍 Search Knowledge Base")
        st.info("Ask questions about your indexed documents for intelligent retrieval.")
        
        rag_query = st.text_area(
            "Ask a question about your documents",
            placeholder="What would you like to know?",
            height=100,
            label_visibility="collapsed"
        )
        
        if st.button("🔎 Search", use_container_width=True):
            if not rag_query.strip(): 
                st.error("Please enter a question.")
            else:
                with st.spinner("Searching knowledge base..."):
                    try:
                        # Initialize data from disk if not already done
                        initialize_data(force_reload=False)
                        
                        # Check if vector store has documents
                        if not st.session_state.data_initialized:
                            st.warning("⚠️ No documents indexed yet. Please upload and index a document first.")
                            logger.warning("No documents in vector store")
                            st.stop()
                        
                        # Retrieve relevant documents using RAG service
                        results = rag_service.retrieve_documents(rag_query)
                        
                        if not results:
                            logger.error("No relevant info found in knowledge database")
                            st.warning("⚠️ No relevant information found. Try uploading more documents or refining your query.")
                        else:
                            st.success(f"✅ Found {len(results)} relevant documents")
                            
                            # Extract context using RAG service
                            context = rag_service.extract_context(results)
                            
                            # Generate answer using LLM service
                            response = llm_service.answer_question(rag_query, context)
                            st.markdown(response)
                    except Exception as e:
                        st.error(f"❌ Error during search: {str(e)}")
                        logger.error(f"Search error: {str(e)}")
                        import traceback
                        traceback.print_exc()

# ==================== TAB 3: RECRUITER AGENT ====================
with tab_3:
    st.header("📧 Recruiter Agent")
    st.info("Upload a CSV file with recruiter information to send personalized professional emails.")
    
    col_upload, col_preview = st.columns(2, gap="large")
    
    with col_upload:
        st.subheader("📤 CSV Upload")
        st.markdown("**Required CSV columns:**")
        st.markdown("- `name`: Recruiter's name")
        st.markdown("- `email`: Recruiter's email address")
        st.markdown("- `job_description`: Job description to customize email")
        
        uploaded_csv = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            label_visibility="collapsed"
        )
        
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                st.session_state.recruiter_df = df
                st.success(f"✅ CSV loaded: {uploaded_csv.name}")
                st.metric("Total Records", len(df))
                
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
                logger.error(f"CSV reading error: {str(e)}")
    
    with col_preview:
        st.subheader("👁️ Data Preview")
        if "recruiter_df" in st.session_state:
            df = st.session_state.recruiter_df
            
            # Validate required columns
            required_cols = ["name", "email", "job_description"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show data types
                st.subheader("Column Info")
                st.dataframe(pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Non-Null": df.count()
                }), use_container_width=True)
    
    # Processing section
    st.divider()
    st.subheader("⚙️ Email Generation & Sending")
    
    if "recruiter_df" in st.session_state:
        df = st.session_state.recruiter_df
        required_cols = ["name", "email", "job_description"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Cannot proceed. Missing columns: {', '.join(missing_cols)}")
        else:
            col_config, col_action = st.columns([2, 1], gap="large")
            
            with col_config:
                st.markdown("**Processing Settings**")
                batch_size = st.number_input(
                    "Batch size (emails per batch)",
                    min_value=1,
                    max_value=len(df),
                    value=min(5, len(df)),
                    step=1
                )
                send_all = st.checkbox(
                    "Send all emails",
                    value=False,
                    help="If unchecked, will process only the first batch"
                )
            
            with col_action:
                st.markdown("**Action**")
                st.markdown("")
                if st.button("🚀 Send Emails", use_container_width=True, key="send_emails_btn"):
                    if df.empty:
                        st.error("❌ No data to process")
                    else:
                        try:
                            progress_bar = st.progress(0, text="Starting email processing...")
                            status_placeholder = st.empty()
                            
                            # Determine how many records to process
                            records_to_process = len(df) if send_all else min(batch_size, len(df))
                            df_to_process = df.iloc[:records_to_process].reset_index(drop=True)
                            
                            # Use the processing function from recruiter_agent
                            with st.spinner("Processing emails..."):
                                processing(df_to_process)
                            
                            progress_bar.progress(1.0, text="✅ Processing complete!")
                            
                            # Display summary
                            st.divider()
                            st.subheader("📊 Processing Summary")
                            summary_col1, summary_col2 = st.columns(2)
                            
                            with summary_col1:
                                st.metric("📨 Total Processed", records_to_process)
                            with summary_col2:
                                st.metric("📧 Batch Size Used", batch_size)
                            
                            st.success(f"✅ Successfully sent {records_to_process} emails!")
                            st.info("All emails have been sent. Check your email logs for details.")
                        
                        except Exception as e:
                            logger.error(f"Unfortunatly Emails are not sended: {e} ")
                            raise Exception(f"Unfortunatly Emails are not sended: {e}") from e
    else:
        st.warning("⚠️ Please upload a CSV file first to proceed.")
