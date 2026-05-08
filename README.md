# RAG Chatbot - Chat with your PDF

A production-ready Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about its content. Built with LangChain, Google Gemini, ChromaDB, and Streamlit.

## Features

- PDF Upload through a clean web interface
- Smart retrieval using vector embeddings
- Conversational chat-style UI with message history
- Fast responses powered by Google Gemini 2.5 Flash
- Local vector store with ChromaDB

## Tech Stack

- LLM: Google Gemini 2.5 Flash
- Framework: LangChain
- Embeddings: Gemini Embedding 001
- Vector Database: ChromaDB
- Frontend: Streamlit
- PDF Processing: PyPDF
- Language: Python 3.12

## Architecture

PDF Upload -> Text Splitter -> Embeddings -> ChromaDB
User Question -> Retriever -> Top-K Chunks -> Gemini LLM -> Answer

## Getting Started

### Prerequisites

- Python 3.12+
- Google Gemini API key (get one at https://aistudio.google.com/apikey)

### Installation

1. Clone the repository:
   git clone https://github.com/avinashbolligarla/rag-chatbot.git
   cd rag-chatbot

2. Create and activate a virtual environment:
   python3.12 -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Create a .env file in the project root and add your Gemini API key:
   GOOGLE_API_KEY=your_api_key_here

### Running the App

Web UI (recommended):
   streamlit run app.py

The app will open at http://localhost:8501

CLI version:
   python rag.py

## Project Structure

- app.py - Streamlit web interface
- rag.py - CLI version of the chatbot
- requirements.txt - Python dependencies
- .env - API keys (not committed)
- .gitignore - Git ignore rules
- README.md - This file

## How It Works

1. Document Loading - PDF is parsed using PyPDFLoader
2. Text Chunking - Document is split into 500-character chunks with 50-character overlap
3. Embedding Generation - Each chunk is converted to a vector using Gemini's embedding model
4. Vector Storage - Embeddings are stored in ChromaDB for fast similarity search
5. Retrieval - When a question is asked, top 3 most relevant chunks are retrieved
6. Generation - Retrieved context plus question are sent to Gemini, which generates an answer

## Future Enhancements

- Support for multiple file formats (DOCX, TXT, MD)
- Multi-document chat
- Conversation memory across sessions
- Source citations with page numbers
- Deployment to Streamlit Cloud

## Author

Avinash Bolligarla
- MS in IT Management, Webster University
- LinkedIn: linkedin.com/in/avinashbolligarla
- GitHub: github.com/avinashbolligarla

## License

This project is open source and available under the MIT License.
