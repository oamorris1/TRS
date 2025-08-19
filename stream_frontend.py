import streamlit as st
import requests
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any

# Configure the app
st.set_page_config(
    page_title="Financial Due Diligence Assistant",
    page_icon="💰",
    layout="wide"
)

# API endpoint configuration
API_URL = "http://localhost:8000"  # Change to FastAPI server URL in azure webapp after we push it

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "citations" not in st.session_state:
    st.session_state.citations = []

if "available_models" not in st.session_state:
    # get available models from API
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            models_data = response.json()
            st.session_state.available_models = models_data.get("models", ["gpt-4o", "gpt4o-mini"])
            st.session_state.default_model = models_data.get("default", "gpt-4o")
        else:
            # Fallback if API call fails
            st.session_state.available_models = ["gpt-4o", "gpt4o-mini"]
            st.session_state.default_model = "gpt-4o"
    except Exception as e:
        st.error(f"Could not fetch available models: {str(e)}")
        st.session_state.available_models = ["gpt-4o", "gpt4o-mini"]
        st.session_state.default_model = "gpt-4o"

if "selected_model" not in st.session_state:
    st.session_state.selected_model = st.session_state.default_model

def extract_citations_from_response(response_text: str) -> List[int]:
    """Extract [docX] citations from response text"""
    doc_pattern = re.compile(r'\[doc(\d+)\]')
    doc_matches = doc_pattern.findall(response_text)
    return sorted(set([int(m) for m in doc_matches]))

def query_api(question, session_id, model):
    """Send a query to the FastAPI backend"""
    try:
        payload = {
            "query": question,
            "session_id": session_id,
            "model": model 
        }
        
        response = requests.post(
            f"{API_URL}/chat",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
            return {"answer": f"Error: Could not get response from API (Status: {response.status_code})"}
    except Exception as e:
        st.error(f"Exception when calling API: {str(e)}")
        return {"answer": f"Error: {str(e)}"}

# App title and description
st.title("💰 Financial Due Diligence Assistant")
st.subheader("AI-powered investment analysis and company research")

# Quick analysis buttons
st.markdown("### Quick Analysis Templates")
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("📊 Financial Health Check", use_container_width=True):
        st.session_state.template_query = "Analyze the financial health of this company including liquidity ratios, debt levels, profitability trends, and cash flow analysis."

with col_b:
    if st.button("🎯 Investment Risks", use_container_width=True):
        st.session_state.template_query = "Identify key investment risks including market risks, operational risks, financial risks, and regulatory challenges."

with col_c:
    if st.button("💡 Growth Analysis", use_container_width=True):
        st.session_state.template_query = "Analyze revenue growth trends, market expansion opportunities, competitive positioning, and future growth prospects."

# Create layout with two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "model" in message and message["role"] == "assistant":
                st.caption(f"Model: {message['model']} | {message.get('timestamp', 'N/A')}")

# Right column - Citations and Sources Panel
with col2:
    st.markdown("### 📚 Document Sources")
    
    # Settings section
    with st.expander("⚙️ Settings", expanded=False):
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=st.session_state.available_models,
            index=st.session_state.available_models.index(st.session_state.selected_model)
        )
        
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
        
        # Session info
        st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
        
        # Reset conversation
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.citations = []
            st.success("Conversation has been reset!")
            st.rerun()
    
    # Analysis Tools
    with st.expander("🔧 Analysis Tools", expanded=False):
        st.markdown("**Document Types:**")
        st.markdown("• 10-K Annual Reports")
        st.markdown("• 10-Q Quarterly Reports") 
        st.markdown("• 8-K Current Reports")
        st.markdown("• Financial Statements")
        st.markdown("• Proxy Statements")
        st.markdown("• Credit Reports")
        
        if st.button("📈 Market Comparison", use_container_width=True):
            st.session_state.template_query = "Compare this company's financial metrics to industry peers and market benchmarks."
        
        if st.button("🔍 Red Flags Analysis", use_container_width=True):
            st.session_state.template_query = "Identify potential red flags in financial statements, accounting practices, and business operations."
    
    # Citations Display
    st.markdown("---")
    
    # Get the latest response to extract citations from
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
        latest_response = st.session_state.chat_history[-1]["content"]
        cited_docs = extract_citations_from_response(latest_response)
        
        if cited_docs:
            st.markdown("**📄 Referenced Documents:**")
            for doc_num in cited_docs:
                # Get the actual source data
                source_data = None
                if hasattr(st.session_state, 'citations') and st.session_state.citations:
                    for source in st.session_state.citations:
                        if source.get('doc_id') == f"doc{doc_num}":
                            source_data = source
                            break
                
                with st.expander(f"📄 [doc{doc_num}] - {source_data.get('title', 'Document') if source_data else 'Document'}", expanded=False):
                    if source_data:
                        # Clean, essential information only
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.markdown(f"**Document Type:** {source_data.get('document_type', 'N/A')}")
                            st.markdown(f"**File Name:** {source_data.get('title', 'N/A')}")
                            
                            # Extract clean file path if available
                            storage_path = source_data.get('metadata', {}).get('metadata_storage_path', 'N/A')
                            if storage_path != 'N/A':
                                # Show just the filename part
                                file_name = storage_path.split('/')[-1] if '/' in storage_path else storage_path
                                st.markdown(f"**Source:** {file_name}")
                        
                        with col_b:
                            st.metric("Relevance Score", f"{source_data.get('relevance_score', 0):.2f}")
                            st.metric("Content Length", f"{source_data.get('content_length', 0):,} chars")
                        
                        # Show chunk content (the actual retrieved text)
                        st.markdown("**Retrieved Content:**")
                        chunk_content = source_data.get('content_preview', source_data.get('full_content', ''))
                        if len(chunk_content) > 500:
                            chunk_content = chunk_content[:500] + "..."
                        
                        st.text_area(
                            "Chunk used for analysis:", 
                            chunk_content, 
                            height=120, 
                            key=f"chunk_{doc_num}",
                            disabled=True
                        )
                    else:
                        st.warning("Source data not available")
        else:
            st.info("💡 Citations from your analysis will appear here")
    else:
        st.info("💡 Ask a question to see document sources")
    
    # Quick insights section
    with st.expander("💡 Quick Insights", expanded=True):
        insights = [
            "🎯 Focus on cash flow trends",
            "📊 Check debt-to-equity ratios", 
            "🔄 Review working capital cycles",
            "📈 Analyze revenue quality",
            "⚠️ Monitor risk factors"
        ]
        
        for insight in insights:
            st.markdown(f"• {insight}")

# Chat input OUTSIDE columns - keeps it at the bottom
user_input = st.chat_input("Ask about financial analysis, SEC filings, company valuation, or investment research...")

# Handle template queries
if "template_query" in st.session_state:
    user_input = st.session_state.template_query
    del st.session_state.template_query

# Process input OUTSIDE columns
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Get the response from the API
    with st.spinner("Analyzing financial data..."):
        response = query_api(
            user_input, 
            st.session_state.session_id, 
            st.session_state.selected_model
        )
        
        answer = response.get("answer", "Sorry, I couldn't process your question.")
        
        # Extract and store citations
        cited_docs = extract_citations_from_response(answer)
        if "sources" in response:
            st.session_state.citations = response["sources"]
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response.get("answer", "Sorry, I couldn't process your question."),
        "model": response.get("model", st.session_state.selected_model),
        "timestamp": response.get("timestamp", datetime.now().isoformat())
    })
    
    # Rerun to display new messages
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Financial Due Diligence Assistant** - Powered by Azure OpenAI and Azure AI Search")