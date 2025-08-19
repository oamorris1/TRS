import sys
import subprocess
import os
from config import Config

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import fastapi
        import streamlit
        import langchain
        import openai
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def validate_environment():
    """Validate environment configuration"""
    errors = Config.validate_config()
    
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   • {error}")
        print("\nPlease check your .env file and ensure all required variables are set.")
        return False
    
    print("✅ Environment configuration is valid")
    return True

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", 
        "backend_app:app", 
        "--host", Config.API_HOST,
        "--port", str(Config.API_PORT),
        "--reload" if Config.DEBUG else "--no-reload"
    ]
    
    return subprocess.Popen(backend_cmd)

def start_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "stream_frontend.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501"
    ]
    
    return subprocess.Popen(frontend_cmd)

def main():
    """Main startup function"""
    print("🏦 Financial Due Diligence Assistant - Starting Up...")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    print("\n📋 Configuration Summary:")
    print(f"   • Azure OpenAI Endpoint: {Config.AZURE_OPENAI_ENDPOINT}")
    print(f"   • Azure Search Endpoint: {Config.AZURE_SEARCH_ENDPOINT}")
    print(f"   • Search Index: {Config.AZURE_SEARCH_INDEX}")
    print(f"   • Available Models: {list(Config.MODEL_DEPLOYMENTS.keys())}")
    print(f"   • API Port: {Config.API_PORT}")
    
    try:
        # Start backend
        backend_process = start_backend()
        
        # Wait a moment for backend to start
        import time
        time.sleep(3)
        
        # Start frontend  
        frontend_process = start_frontend()
        
        print("\n" + "=" * 60)
        print("🎉 Application started successfully!")
        print(f"   • Backend API: http://localhost:{Config.API_PORT}")
        print(f"   • Frontend UI: http://localhost:8501")
        print(f"   • API Health Check: http://localhost:{Config.API_PORT}/health")
        print(f"   • API Documentation: http://localhost:{Config.API_PORT}/docs")
        print("\nPress Ctrl+C to stop the application")
        print("=" * 60)
        
        # Wait for processes
        try:
            backend_process.wait()
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down application...")
            backend_process.terminate()
            frontend_process.terminate()
            print("✅ Application stopped successfully")
    
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()