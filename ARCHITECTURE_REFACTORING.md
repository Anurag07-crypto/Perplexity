# Architecture Refactoring: Service Layer Separation

## Problem Identified
The original `frontend.py` mixed business logic with UI code, specifically:
- **Line 304-358**: LLM prompt construction embedded in Streamlit UI
- **Line 339-348**: Manual context extraction logic
- **Line 359**: Direct LLM invocation in the UI layer

This violates the **Separation of Concerns** principle and makes the code:
- Hard to test (business logic cannot be unit tested independently)
- Hard to reuse (logic tied to Streamlit UI)
- Hard to maintain (changes to prompt format require UI file modification)

## Solution Implemented

### 1. New Service Layer: `frontend_server/services.py`

#### `RAGService` Class
**Responsibility:** Manage all RAG (Retrieval-Augmented Generation) operations

Methods:
- `retrieve_documents(query)` - Retrieve relevant documents from vector store
- `extract_context(documents)` - Extract text content from various document formats
- Handles polymorphic document types (dict or LangChain Document objects)

**Benefits:**
- Centralized document retrieval logic
- Handles document type polymorphism in one place
- Fully testable without Streamlit

#### `LLMService` Class
**Responsibility:** Manage all LLM interactions and prompt engineering

Methods:
- `build_rag_prompt(query, context)` - Construct well-formatted prompts
- `invoke_llm(prompt)` - Execute LLM with proper error handling
- `answer_question(query, context)` - End-to-end question answering pipeline

**Benefits:**
- Prompt template is centralized and versioned
- Can easily swap models or change prompt format
- LLM logic completely independent of UI framework
- Easy to test with different prompts and contexts

### 2. Updated `frontend.py` (UI Layer)

**Before (Mixed Logic):**
```python
# Line 304-359: Business logic in UI
context_parts = []
for doc in results:
    if isinstance(doc, dict):
        content = doc.get('content') or doc.get('document', '')
    # ... more extraction logic
PROMPT = f'''### You are a 'Personal Assitant'...'''
response = llm.invoke(PROMPT)
st.markdown(response.content)
```

**After (Clean Separation):**
```python
# Service handles all logic
context = rag_service.extract_context(results)
response = llm_service.answer_question(rag_query, context)
st.markdown(response)
```

### 3. Benefits of This Architecture

| Aspect | Before | After |
|--------|--------|-------|
| **Testability** | Cannot unit test business logic | Full unit test coverage possible |
| **Reusability** | Tied to Streamlit UI | Services can be used in FastAPI, CLI, or other UIs |
| **Maintainability** | Prompt buried in frontend code | Centralized prompt management |
| **Error Handling** | Mixed with UI error handling | Dedicated business logic error handling |
| **Consistency** | Prompt format varies | Standardized prompt templates |

## File Structure

```
frontend_server/
├── frontend.py          # UI layer only (Streamlit)
├── services.py          # Business logic layer
└── __init__.py
```

## Usage Example

```python
from frontend_server.services import RAGService, LLMService

# Initialize services
rag_service = RAGService(embedding_manager, vector_store)
llm_service = LLMService()

# Use in UI
results = rag_service.retrieve_documents("user query")
context = rag_service.extract_context(results)
answer = llm_service.answer_question("user query", context)
```

## Future Improvements

1. **Add Configuration Management**: Move prompt templates to config files
2. **Add Caching Layer**: Cache embeddings and retrieval results
3. **Add Monitoring**: Log service metrics (retrieval latency, LLM response time)
4. **Add Validation**: Input validation in services before LLM calls
5. **Create Abstract Base Classes**: For easy extension and alternative implementations

## Testing Strategy

The new architecture enables:
```python
# Unit test example
def test_rag_service():
    service = RAGService(mock_embedding_manager, mock_vector_store)
    results = service.extract_context([mock_document])
    assert "expected content" in results

def test_llm_service():
    service = LLMService()
    prompt = service.build_rag_prompt("What?", "Context here")
    assert "Personal Assistant" in prompt
    assert "Context here" in prompt
```
