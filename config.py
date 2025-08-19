import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Azure OpenAI Settings
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") 
    AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-09-01-preview")
    
    # Azure AI Search Settings
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
    AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "financial-docs-index")
    AZURE_SEARCH_SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "default")
    
    # Model Deployments
    MODEL_DEPLOYMENTS = {
        "gpt-4o": os.getenv("GPT_4O_DEPLOYMENT", "gpt-4o"),
        "gpt4o-mini": os.getenv("GPT_4O_MINI_DEPLOYMENT", "gpt4o-mini")
    }
    
    # Text Embedding
    TEXT_EMBEDDING_DEPLOYMENT = os.getenv("TEXT_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Financial Analysis Settings
    RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 8))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Document Types for Financial Analysis
    SUPPORTED_DOCUMENT_TYPES = [
        "10-K Annual Report",
        "10-Q Quarterly Report", 
        "8-K Current Report",
        "Financial Statement",
        "Proxy Statement",
        "Credit Report",
        "Earnings Release",
        "Analyst Report"
    ]
    
    # Financial Metrics Categories
    FINANCIAL_METRICS = {
        "Profitability": [
            "Gross Profit Margin",
            "Operating Margin",
            "Net Profit Margin", 
            "Return on Assets (ROA)",
            "Return on Equity (ROE)",
            "Return on Invested Capital (ROIC)"
        ],
        "Liquidity": [
            "Current Ratio",
            "Quick Ratio", 
            "Cash Ratio",
            "Working Capital",
            "Cash Conversion Cycle"
        ],
        "Leverage": [
            "Debt-to-Equity Ratio",
            "Debt-to-Assets Ratio",
            "Interest Coverage Ratio",
            "EBITDA Coverage Ratio",
            "Debt Service Coverage Ratio"
        ],
        "Efficiency": [
            "Asset Turnover",
            "Inventory Turnover",
            "Receivables Turnover", 
            "Days Sales Outstanding",
            "Days Payable Outstanding"
        ],
        "Valuation": [
            "Price-to-Earnings (P/E)",
            "Price-to-Book (P/B)",
            "Enterprise Value/EBITDA",
            "Price/Sales Ratio",
            "PEG Ratio"
        ]
    }
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate required configuration values"""
        errors = []
        
        required_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_SEARCH_ENDPOINT", 
            "SEARCH_API_KEY"
        ]
        
        for var in required_vars:
            if not getattr(cls, var):
                errors.append(f"Missing required environment variable: {var}")
        
        return errors
    
    @classmethod
    def get_model_deployment(cls, model_name: str) -> str:
        """Get deployment name for a model"""
        return cls.MODEL_DEPLOYMENTS.get(model_name, "gpt-4o")
