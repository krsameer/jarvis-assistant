#!/bin/bash

# Jarvis AI Assistant - Quick Start Script

echo "================================="
echo "🤖 Jarvis AI Assistant Setup"
echo "================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Please install Ollama first."
    echo "Visit: https://ollama.ai"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your Pinecone API key!"
    echo "   nano .env"
    echo ""
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "================================="
echo "🚀 Next Steps:"
echo "================================="
echo ""
echo "1. Configure your Pinecone API key:"
echo "   nano .env"
echo ""
echo "2. Ensure Ollama is running with ollama3 model:"
echo "   ollama run ollama3"
echo ""
echo "3. In a new terminal, start the backend:"
echo "   cd backend"
echo "   uvicorn main:app --reload"
echo ""
echo "4. In another terminal, start the frontend:"
echo "   cd frontend"
echo "   streamlit run streamlit_app.py"
echo ""
echo "5. Open your browser:"
echo "   Frontend: http://localhost:8501"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "================================="
