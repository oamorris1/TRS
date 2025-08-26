from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Literal
import os
import uuid
import re
from datetime import datetime

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'), override=True)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Due Diligence API", 
    description="AI-powered financial analysis and investment research API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    model: Optional[Literal["gpt-4o", "gpt4o-mini"]] = "gpt-4o"

class DocumentSource(BaseModel):
    doc_id: str
    title: str
    content_preview: str
    document_type: str
    relevance_score: float
    metadata: Dict[str, Any] = {}

# Chat history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a chat history for a session"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Model deployments
MODEL_DEPLOYMENTS = {
    "gpt-4o": "gpt-4o",  # Replace with actual deployment name
    "gpt4o-mini": "gpt4o-mini"  # Replace with actual deployment name
}

def get_llm(model_name: str):
    """Get Azure OpenAI LLM with specified model"""
    deployment_name = MODEL_DEPLOYMENTS.get(model_name, "gpt-4o")
    return AzureChatOpenAI(
        deployment_name=deployment_name,
        model_name=model_name,
        temperature=0,
        streaming=False,
        api_version="2024-09-01-preview"
    )

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002", 
    model="text-embedding-ada-002", 
    chunk_size=10
)

# Connect to Azure AI Search - Update with your financial documents index
index_name = "rag-1754515975021"  # Update with your financial documents index name
vector_store = AzureSearch(
    azure_search_endpoint="add search endpoint",
    azure_search_key="add key",
    index_name=index_name,
    embedding_function=embeddings.embed_query
)

# Create retriever for financial documents
retriever = AzureAISearchRetriever(
    content_key="chunk", 
    top_k=8,  # Increased for more comprehensive financial analysis
    index_name=index_name,
    service_name="ppl-test-aisearch1273293761837", 
    api_key=os.environ.get("SEARCH_API_KEY")
)

# Cache for model-specific RAG chains
rag_chains = {}

def extract_document_sources(documents) -> List[Dict[str, Any]]:
    """Extract and format document sources for citation panel with full chunk details"""
    sources = []
    for i, doc in enumerate(documents, 1):
        # Extract metadata from document - your index uses different field names
        metadata = getattr(doc, 'metadata', {})
        
        # Get the full chunk content - your field is called "chunk"
        content = ""
        if hasattr(doc, 'page_content') and doc.page_content:
            content = doc.page_content
        elif hasattr(doc, 'content') and doc.content:
            content = doc.content
        else:
            content = str(doc)  # Fallback
        
        # Get title from your index structure
        title = metadata.get('title', f'Document {i}')
        
        # Extract document type from title
        doc_type = "Financial Document"
        if "10K" in title or "10-K" in title:
            doc_type = "10-K Annual Report"
        elif "10Q" in title or "10-Q" in title:
            doc_type = "10-Q Quarterly Report"
        elif "Credit" in title:
            doc_type = "Credit Analysis"
        
        # Get search score
        search_score = metadata.get('@search.score', 0.0)
        
       
         # Build source info
        sources.append({
            "doc_id": f"doc{i}",
            "title": title,
            "full_content": content,
            "content_preview": content[:300] + "..." if len(content) > 300 else content,
            "content_length": len(content),
            "document_type": doc_type,
            "relevance_score": search_score,
            "metadata": metadata,
            "azure_metadata": {
                'search_score': search_score,
                'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                'parent_id': metadata.get('parent_id', 'N/A'),
                'company_name': 'TechCorp Incorporated',
                'filing_date': '2024-12-31',
                'fiscal_period': 'FY 2024',
                'section': 'Financial Analysis'
            }
        })
        
    
    return sources

def get_rag_chain(model_name: str):
    """Get or create a RAG chain for financial analysis"""
    if model_name in rag_chains:
        return rag_chains[model_name]
    
    # Initialize model
    llm = get_llm(model_name)
    
    # Context-aware question reformulation for financial queries
    contextualize_q_system_prompt = (
        "You are FinanceExpert, specialized in financial due diligence and investment analysis. "
        "Given the chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone, comprehensive question that:"
        "\n\n"
        "1. Captures all financial concepts, ratios, and metrics mentioned"
        "2. Expands financial acronyms (e.g., P/E = Price-to-Earnings, ROE = Return on Equity)"
        "3. Includes related financial terms and synonyms for better retrieval"
        "4. Specifies time periods when relevant (quarterly, annual, year-over-year)"
        "5. Can be understood without the chat history"
        "\n\n"
        "Focus on financial analysis concepts like: profitability, liquidity, solvency, efficiency, "
        "valuation metrics, cash flow analysis, risk assessment, and competitive positioning."
        "\n\n"
        "Do NOT answer the question, just reformulate it to maximize retrieval of relevant financial information."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Financial analysis QA prompt
    system_prompt = (
        "You are FinanceExpert, an AI assistant specialized in financial due diligence, investment analysis, "
        "and company research. You analyze SEC filings, financial statements, and provide investment insights."
        "\n\n"
        "**CORE EXPERTISE AREAS:**"
        "\n"
        "• Financial Statement Analysis (Income Statement, Balance Sheet, Cash Flow)"
        "\n"
        "• Investment Valuation (DCF, Comparable Company Analysis, Precedent Transactions)"
        "\n"
        "• Risk Assessment (Credit Risk, Market Risk, Operational Risk)"
        "\n"
        "• SEC Filings Analysis (10-K, 10-Q, 8-K, Proxy Statements)"
        "\n"
        "• Due Diligence (Management Assessment, Competitive Analysis, Industry Trends)"
        "\n"
        "• Financial Ratios (Profitability, Liquidity, Leverage, Efficiency)"
        "\n\n"
        "**ANALYSIS GUIDELINES:**"
        "\n"
        "1. **COMPREHENSIVE ANALYSIS**: Provide thorough financial analysis with specific metrics and ratios"
        "\n"
        "2. **CONTEXTUAL INSIGHTS**: Explain financial data within industry and market context"
        "\n"
        "3. **RISK IDENTIFICATION**: Highlight potential red flags and investment risks"
        "\n"
        "4. **ACTIONABLE RECOMMENDATIONS**: Provide clear investment implications and next steps"
        "\n"
        "5. **QUANTITATIVE FOCUS**: Use specific numbers, percentages, and financial metrics when available"
        "\n"
        "6. **TREND ANALYSIS**: Identify patterns over multiple periods when data permits"
        "\n\n"
        "**CITATION REQUIREMENTS:**"
        "\n"
        "For each piece of financial information used, cite your source using [docX] format where X is the "
        "1-based index of the document. Examples:"
        "\n"
        "• 'Revenue increased 15% year-over-year to $2.5 billion [doc1]'"
        "\n"
        "• 'The company's debt-to-equity ratio of 0.65 indicates moderate leverage [doc2]'"
        "\n"
        "• 'Management highlighted supply chain risks in their recent 10-K filing [doc3]'"
        "\n\n"
        "**RESPONSE STRUCTURE:**"
        "\n"
        "Start with a brief executive summary, then provide detailed analysis organized by:"
        "\n"
        "- Financial Performance & Trends"
        "\n"
        "- Key Financial Ratios & Metrics"
        "\n"
        "- Risk Factors & Concerns"
        "\n"
        "- Investment Implications"
        "\n\n"
        "If information is incomplete, specify what additional data would be helpful for a complete analysis."
        "\n\n"
        "Retrieved financial documents:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Cache and return
    rag_chains[model_name] = rag_chain
    return rag_chain

async def process_query(query: str, session_id: Optional[str] = None, model: str = "gpt-4o"):
    """Process a financial analysis query using RAG with conversation history"""
    # Validate model selection
    if model not in MODEL_DEPLOYMENTS:
        model = "gpt-4o"
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Get the RAG chain for the model
    rag_chain = get_rag_chain(model)
    
    # Wrap with message history management
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    try:
        # Execute the query
        result = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Extract document sources for citation panel
        documents = result.get("context", [])
        sources = extract_document_sources(documents)
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "session_id": session_id,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        print(f"Error processing financial query: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Unified chat endpoint
@app.api_route("/chat", methods=["GET", "POST"])
async def chat(request: Request):
    """Financial analysis chat endpoint"""
    if request.method == "GET":
        query = request.query_params.get("query")
        session_id = request.query_params.get("session_id")
        model = request.query_params.get("model", "gpt-4o")
    else:
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            body = await request.json()
            query = body.get("query")
            session_id = body.get("session_id")
            model = body.get("model", "gpt-4o")
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form = await request.form()
            query = form.get("query")
            session_id = form.get("session_id")
            model = form.get("model", "gpt-4o")
        else:
            return {"error": "Unsupported request format"}
    
    if not query:
        return {"error": "No query provided"}
        
    if model not in MODEL_DEPLOYMENTS:
        model = "gpt-4o"
    
    result = await process_query(query, session_id, model)
    return result

# Available models endpoint
@app.get("/models")
async def list_models():
    """Return available model options"""
    return {
        "models": list(MODEL_DEPLOYMENTS.keys()),
        "default": "gpt-4o"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Financial Due Diligence API",
        "timestamp": datetime.now().isoformat()
    }

# Clear chat history
@app.post("/clear")
async def clear_history(session_id: str):
    """Clear chat history for a session"""
    if session_id in store:
        store[session_id] = ChatMessageHistory()
        return {"status": "success", "message": f"Chat history cleared for session {session_id}"}
    return {"status": "error", "message": f"Session {session_id} not found"}

# Get detailed chunk information
@app.get("/chunk/{session_id}/{doc_id}")
async def get_chunk_details(session_id: str, doc_id: str):
    """Get detailed information about a specific chunk/document"""
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # This would typically retrieve from a cache or re-query the vector store
    # For now, return placeholder information
    return {
        "doc_id": doc_id,
        "session_id": session_id,
        "message": "Detailed chunk information would be retrieved here",
        "note": "Implement chunk caching or vector store re-query for full functionality"
    }

# Enhanced search endpoint for debugging
@app.post("/debug/search")
async def debug_search(query: str, top_k: int = 5):
    """Debug endpoint to see raw search results from Azure AI Search"""
    try:
        # Perform direct search
        docs = retriever.get_relevant_documents(query)
        
        debug_results = []
        for i, doc in enumerate(docs[:top_k], 1):
            debug_results.append({
                "doc_number": i,
                "content": doc.page_content,
                "content_length": len(doc.page_content),
                "metadata": getattr(doc, 'metadata', {}),
                "score": getattr(doc, 'score', 0.0),
                "type": str(type(doc)),
                "attributes": [attr for attr in dir(doc) if not attr.startswith('_')]
            })
        
        return {
            "query": query,
            "total_results": len(docs),
            "showing": len(debug_results),
            "results": debug_results
        }
    
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "query": query
        }

# Document analysis endpoint
@app.post("/analyze")
async def analyze_document(document_type: str, company_symbol: Optional[str] = None):
    """Analyze specific document types"""
    analysis_prompts = {
        "10k": f"Provide a comprehensive analysis of the 10-K filing for {company_symbol or 'this company'}, focusing on business overview, risk factors, financial performance, and key metrics.",
        "10q": f"Analyze the most recent 10-Q filing for {company_symbol or 'this company'}, highlighting quarterly performance, changes from previous quarters, and management guidance.",
        "financial_health": f"Conduct a financial health assessment for {company_symbol or 'this company'} including liquidity, solvency, profitability, and efficiency metrics.",
        "investment_risks": f"Identify and analyze key investment risks for {company_symbol or 'this company'} including market, operational, financial, and regulatory risks."
    }
    
    if document_type not in analysis_prompts:
        raise HTTPException(status_code=400, detail="Invalid document type")
    
    query = analysis_prompts[document_type]
    result = await process_query(query)
    return result
# Enhanced search endpoint for debugging
@app.get("/debug/search")
async def debug_search(query: str, top_k: int = 5):
    """Debug endpoint to see raw search results from Azure AI Search"""
    try:
        # Perform direct search
        docs = retriever.get_relevant_documents(query)
        
        debug_results = []
        for i, doc in enumerate(docs[:top_k], 1):
            debug_results.append({
                "doc_number": i,
                "content": getattr(doc, 'page_content', 'No content'),
                "content_length": len(getattr(doc, 'page_content', '')),
                "metadata": getattr(doc, 'metadata', {}),
                "score": getattr(doc, 'score', 0.0),
                "type": str(type(doc)),
                "attributes": [attr for attr in dir(doc) if not attr.startswith('_')]
            })
        
        return {
            "query": query,
            "total_results": len(docs),
            "showing": len(debug_results),
            "results": debug_results
        }
    
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "query": query
        }
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
