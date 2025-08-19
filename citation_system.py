import streamlit as st
import requests
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

class CitationManager:
    """Manages document citations and source tracking"""
    
    def __init__(self):
        self.citations = {}
        self.source_metadata = {}
    
    def extract_citations(self, response_text: str) -> List[int]:
        """Extract [docX] citations from response text"""
        doc_pattern = re.compile(r'\[doc(\d+)\]')
        doc_matches = doc_pattern.findall(response_text)
        return sorted(set([int(m) for m in doc_matches]))
    
    def add_sources(self, sources: List[Dict[str, Any]]):
        """Add source documents to the citation manager"""
        for source in sources:
            doc_id = source.get('doc_id')
            if doc_id:
                self.source_metadata[doc_id] = source
    
    def get_citation_details(self, doc_num: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a cited document"""
        doc_id = f"doc{doc_num}"
        return self.source_metadata.get(doc_id)
    
    def format_citation_display(self, doc_num: int) -> str:
        """Format citation for display in the UI"""
        details = self.get_citation_details(doc_num)
        if not details:
            return f"Document {doc_num}"
        
        doc_type = details.get('document_type', 'Financial Document')
        title = details.get('title', f'Document {doc_num}')
        return f"[{doc_num}] {doc_type}: {title}"

class FinancialAnalysisUI:
    """Enhanced UI components for financial analysis"""
    
    @staticmethod
    def render_financial_metrics_panel():
        """Render a panel with key financial metrics to analyze"""
        st.markdown("### 📊 Key Financial Metrics")
        
        metrics_categories = {
            "Profitability": [
                "Gross Profit Margin",
                "Operating Margin", 
                "Net Profit Margin",
                "Return on Assets (ROA)",
                "Return on Equity (ROE)"
            ],
            "Liquidity": [
                "Current Ratio",
                "Quick Ratio",
                "Cash Ratio",
                "Working Capital"
            ],
            "Leverage": [
                "Debt-to-Equity Ratio",
                "Debt-to-Assets Ratio",
                "Interest Coverage Ratio",
                "EBITDA Coverage Ratio"
            ],
            "Efficiency": [
                "Asset Turnover",
                "Inventory Turnover",
                "Receivables Turnover",
                "Days Sales Outstanding"
            ],
            "Valuation": [
                "Price-to-Earnings (P/E)",
                "Price-to-Book (P/B)",
                "Enterprise Value/EBITDA",
                "Price/Sales Ratio"
            ]
        }
        
        selected_category = st.selectbox(
            "Select Metric Category:",
            list(metrics_categories.keys())
        )
        
        if selected_category:
            st.markdown(f"**{selected_category} Metrics:**")
            for metric in metrics_categories[selected_category]:
                if st.button(f"📈 Analyze {metric}", key=f"metric_{metric}"):
                    query = f"Analyze the {metric} for this company, including current values, historical trends, and industry comparison."
                    st.session_state.template_query = query
    
    @staticmethod
    def render_document_type_selector():
        """Render document type analysis options"""
        st.markdown("### 📋 Document Analysis")
        
        doc_types = {
            "10-K Annual Report": "Analyze the annual report (10-K) focusing on business overview, risk factors, and financial performance.",
            "10-Q Quarterly Report": "Review the quarterly report (10-Q) for recent financial performance and management commentary.",
            "8-K Current Report": "Examine current reports (8-K) for material events and corporate changes.",
            "Financial Statements": "Deep dive into financial statements including income statement, balance sheet, and cash flow.",
            "Proxy Statement": "Analyze proxy statements for executive compensation, governance, and shareholder matters.",
            "Credit Analysis": "Perform credit analysis including debt capacity, credit risk, and rating implications."
        }
        
        selected_doc = st.selectbox(
            "Select Document Type:",
            list(doc_types.keys())
        )
        
        if st.button(f"🔍 Analyze {selected_doc}", key=f"doc_analysis_{selected_doc}"):
            st.session_state.template_query = doc_types[selected_doc]
    
    @staticmethod
    def render_investment_scenarios():
        """Render investment scenario analysis options"""
        st.markdown("### 🎯 Investment Scenarios")
        
        scenarios = {
            "Acquisition Due Diligence": "Conduct comprehensive due diligence for potential acquisition including valuation, synergies, and risk assessment.",
            "IPO Analysis": "Analyze company readiness for IPO including financial metrics, market positioning, and valuation.",
            "Credit Investment": "Evaluate credit investment opportunity including debt capacity, cash flow coverage, and default risk.",
            "Growth Investment": "Assess growth investment potential including market opportunity, competitive position, and scalability.",
            "Distressed Investment": "Analyze distressed investment opportunity including restructuring potential and recovery scenarios.",
            "ESG Investment": "Evaluate Environmental, Social, and Governance factors for sustainable investing."
        }
        
        for scenario, description in scenarios.items():
            if st.button(f"🎯 {scenario}", key=f"scenario_{scenario}"):
                st.session_state.template_query = description

def render_enhanced_citations_panel(citation_manager: CitationManager):
    """Render enhanced citations panel with detailed source information and full chunks"""
    
    st.markdown("### 📚 Document Sources & Citations")
    
    # Get the latest response to extract citations
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
        latest_response = st.session_state.chat_history[-1]["content"]
        cited_docs = citation_manager.extract_citations(latest_response)
        
        if cited_docs:
            st.markdown(f"**📄 {len(cited_docs)} Documents Referenced:**")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📋 Citation List", "📄 Full Chunks", "📊 Search Analytics"])
            
            with tab1:
                for doc_num in cited_docs:
                    details = citation_manager.get_citation_details(doc_num)
                    
                    with st.expander(citation_manager.format_citation_display(doc_num), expanded=False):
                        if details:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Type:** {details.get('document_type', 'N/A')}")
                                st.markdown(f"**Title:** {details.get('title', 'N/A')}")
                                st.markdown(f"**Search Score:** {details.get('relevance_score', 0):.3f}")
                                st.markdown(f"**Content Length:** {details.get('content_length', 0):,} characters")
                                
                                # Show Azure AI Search metadata
                                azure_meta = details.get('azure_metadata', {})
                                if azure_meta:
                                    st.markdown("**Azure Search Properties:**")
                                    meta_col1, meta_col2 = st.columns(2)
                                    
                                    with meta_col1:
                                        if azure_meta.get('company_name', 'N/A') != 'N/A':
                                            st.markdown(f"• **Company:** {azure_meta.get('company_name')}")
                                        if azure_meta.get('filing_date', 'N/A') != 'N/A':
                                            st.markdown(f"• **Filing Date:** {azure_meta.get('filing_date')}")
                                        if azure_meta.get('page_number', 'N/A') != 'N/A':
                                            st.markdown(f"• **Page:** {azure_meta.get('page_number')}")
                                    
                                    with meta_col2:
                                        if azure_meta.get('fiscal_period', 'N/A') != 'N/A':
                                            st.markdown(f"• **Period:** {azure_meta.get('fiscal_period')}")
                                        if azure_meta.get('section', 'N/A') != 'N/A':
                                            st.markdown(f"• **Section:** {azure_meta.get('section')}")
                                        if azure_meta.get('file_type', 'N/A') != 'N/A':
                                            st.markdown(f"• **File Type:** {azure_meta.get('file_type')}")
                                
                                # Show content preview
                                content_preview = details.get('content_preview', '')
                                if content_preview:
                                    st.markdown("**Content Preview:**")
                                    st.text_area(
                                        "Preview", 
                                        content_preview, 
                                        height=150, 
                                        key=f"preview_{doc_num}",
                                        disabled=True
                                    )
                            
                            with col2:
                                if st.button("🔍 Analyze", key=f"analyze_{doc_num}"):
                                    query = f"Provide a detailed analysis of document {doc_num}, highlighting key financial information and insights."
                                    st.session_state.template_query = query
                                
                                if st.button("📖 Summary", key=f"summary_{doc_num}"):
                                    query = f"Summarize the key points from document {doc_num} and explain its relevance to the investment analysis."
                                    st.session_state.template_query = query
                                
                                if st.button("📄 Full Text", key=f"fulltext_{doc_num}"):
                                    st.session_state[f"show_full_{doc_num}"] = not st.session_state.get(f"show_full_{doc_num}", False)
                        else:
                            st.info(f"Document {doc_num} details not available")
            
            with tab2:
                st.markdown("**📄 Complete Retrieved Chunks**")
                
                for doc_num in cited_docs:
                    details = citation_manager.get_citation_details(doc_num)
                    
                    if details:
                        st.markdown(f"### {citation_manager.format_citation_display(doc_num)}")
                        
                        # Show full chunk metadata
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Search Score", f"{details.get('relevance_score', 0):.3f}")
                        with col2:
                            st.metric("Content Length", f"{details.get('content_length', 0):,}")
                        with col3:
                            azure_meta = details.get('azure_metadata', {})
                            chunk_id = azure_meta.get('chunk_id', 'N/A')
                            st.metric("Chunk ID", chunk_id)
                        
                        # Display full content in expandable section
                        full_content = details.get('full_content', '')
                        if full_content:
                            with st.expander(f"📄 Full Chunk Content ({len(full_content):,} chars)", expanded=False):
                                st.text_area(
                                    "Complete chunk content retrieved from Azure AI Search:",
                                    full_content,
                                    height=400,
                                    key=f"full_content_{doc_num}",
                                    disabled=True
                                )
                                
                                # Add copy button functionality info
                                st.caption("💡 Use Ctrl+A to select all content, then Ctrl+C to copy")
                        
                        # Show original metadata
                        original_metadata = details.get('metadata', {})
                        if original_metadata:
                            with st.expander("🔍 Raw Metadata from Azure AI Search", expanded=False):
                                st.json(original_metadata)
                        
                        st.markdown("---")
                
            with tab3:
                st.markdown("**📊 Search & Retrieval Analytics**")
                
                # Document type distribution
                if citation_manager.source_metadata:
                    doc_types = [details.get('document_type', 'Unknown') 
                               for details in citation_manager.source_metadata.values()]
                    
                    type_counts = pd.Series(doc_types).value_counts()
                    
                    st.markdown("**Document Type Distribution:**")
                    for doc_type, count in type_counts.items():
                        st.markdown(f"• {doc_type}: {count} document(s)")
                    
                    # Search scores analysis
                    st.markdown("**Search Relevance Analysis:**")
                    
                    scores_data = []
                    content_lengths = []
                    
                    for doc_id, details in citation_manager.source_metadata.items():
                        if doc_id.replace('doc', '').isdigit():
                            doc_num = int(doc_id.replace('doc', ''))
                            if doc_num in cited_docs:
                                scores_data.append({
                                    'Document': doc_id,
                                    'Search Score': details.get('relevance_score', 0),
                                    'Content Length': details.get('content_length', 0),
                                    'Document Type': details.get('document_type', 'Unknown')
                                })
                                content_lengths.append(details.get('content_length', 0))
                    
                    if scores_data:
                        df = pd.DataFrame(scores_data)
                        
                        # Show metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_score = df['Search Score'].mean()
                            st.metric("Avg Search Score", f"{avg_score:.3f}")
                        with col2:
                            avg_length = df['Content Length'].mean()
                            st.metric("Avg Chunk Length", f"{avg_length:,.0f}")
                        with col3:
                            total_content = df['Content Length'].sum()
                            st.metric("Total Content", f"{total_content:,}")
                        
                        # Visualizations
                        st.markdown("**Search Score Distribution:**")
                        st.bar_chart(df.set_index('Document')['Search Score'])
                        
                        st.markdown("**Content Length vs Search Score:**")
                        st.scatter_chart(df, x='Content Length', y='Search Score', color='Document Type')
                        
                        # Show detailed table
                        st.markdown("**Detailed Retrieval Results:**")
                        st.dataframe(df, use_container_width=True)
                        
                        # Azure AI Search insights
                        st.markdown("**🔍 Retrieval Insights:**")
                        high_relevance = df[df['Search Score'] > 0.8]
                        if not high_relevance.empty:
                            st.success(f"✅ {len(high_relevance)} high-relevance documents (score > 0.8)")
                        
                        low_relevance = df[df['Search Score'] < 0.3]
                        if not low_relevance.empty:
                            st.warning(f"⚠️ {len(low_relevance)} low-relevance documents (score < 0.3)")
                        
                        # Content distribution
                        if content_lengths:
                            avg_length = sum(content_lengths) / len(content_lengths)
                            if avg_length > 2000:
                                st.info("💡 Large chunks detected - consider chunk size optimization")
                            elif avg_length < 200:
                                st.info("💡 Small chunks detected - might need larger chunk sizes")
        else:
            st.info("💡 No citations found in the current response")
            st.markdown("Citations will appear here when you ask questions that reference specific documents.")
    else:
        st.info("💡 Ask a financial analysis question to see document sources")
        
        # Show example queries
        st.markdown("**Example Questions:**")
        examples = [
            "What is the company's debt-to-equity ratio trend?",
            "Analyze the revenue growth over the past 3 years",
            "What are the main risk factors mentioned in the 10-K?",
            "Compare the company's profitability to industry peers"
        ]
        
        for example in examples:
            if st.button(f"💡 {example}", key=f"example_{hash(example)}"):
                st.session_state.template_query = example

def main():
    """Main function to demonstrate the enhanced citation system"""
    st.set_page_config(
        page_title="Financial Analysis with Enhanced Citations",
        page_icon="💰",
        layout="wide"
    )
    
    # Initialize citation manager in session state
    if 'citation_manager' not in st.session_state:
        st.session_state.citation_manager = CitationManager()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.title("💰 Financial Due Diligence with Enhanced Citations")
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Financial Analysis Chat")
        
        # Demo chat history with citations
        demo_response = """
        Based on the financial documents, the company shows strong profitability metrics [doc1]. 
        The debt-to-equity ratio of 0.45 indicates conservative leverage [doc2], while revenue 
        growth of 15% year-over-year demonstrates solid market performance [doc3]. However, 
        the 10-K filing highlights supply chain risks as a key concern [doc4].
        """
        
        st.session_state.chat_history = [
            {"role": "user", "content": "Analyze the company's financial health"},
            {"role": "assistant", "content": demo_response}
        ]
        
        # Add demo sources to citation manager
        demo_sources = [
            {
                "doc_id": "doc1",
                "title": "Q3 2024 Financial Statements",
                "content_preview": "Revenue: $2.5B (+15% YoY), Net Income: $450M (+12% YoY), Gross Margin: 65%...",
                "document_type": "Financial Statement",
                "relevance_score": 0.95
            },
            {
                "doc_id": "doc2", 
                "title": "2024 Annual Report (10-K)",
                "content_preview": "Total Debt: $1.2B, Shareholders' Equity: $2.7B, Debt-to-Equity: 0.45...",
                "document_type": "10-K Annual Report",
                "relevance_score": 0.88
            },
            {
                "doc_id": "doc3",
                "title": "Q3 2024 Earnings Release", 
                "content_preview": "Revenue increased 15% year-over-year driven by strong demand in core markets...",
                "document_type": "8-K Current Report",
                "relevance_score": 0.92
            },
            {
                "doc_id": "doc4",
                "title": "Risk Factors - 2024 10-K",
                "content_preview": "Supply chain disruptions could materially impact our operations and financial results...",
                "document_type": "10-K Annual Report", 
                "relevance_score": 0.78
            }
        ]
        
        st.session_state.citation_manager.add_sources(demo_sources)
        
        # Display chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    with col2:
        # Render enhanced citations panel
        render_enhanced_citations_panel(st.session_state.citation_manager)
        
        st.markdown("---")
        
        # Financial analysis tools
        FinancialAnalysisUI.render_financial_metrics_panel()
        
        with st.expander("📋 Document Analysis", expanded=False):
            FinancialAnalysisUI.render_document_type_selector()
        
        with st.expander("🎯 Investment Scenarios", expanded=False):
            FinancialAnalysisUI.render_investment_scenarios()

if __name__ == "__main__":
    main()