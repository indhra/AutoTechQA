# AutoTechQA

AutoTechQA is an advanced AI-powered question-answering system designed to assist users in finding accurate and relevant information from automotive truck and bus technical manuals. Built using RAG (Retrieval-Augmented Generation) architecture, AutoTechQA leverages state-of-the-art natural language processing and vector search to understand complex queries and provide precise answers from technical documentation.

## Features

- 🚛 **Truck & Bus Manual QA**: Specialized in automotive technical documentation
- 🔍 **Semantic Search**: Uses vector embeddings for intelligent document retrieval
- 🤖 **GPT-4 Powered**: Leverages OpenAI's GPT-4 for accurate answer generation
- 💾 **Vector Database**: ChromaDB for efficient similarity search
- 🎨 **User-Friendly Interface**: Streamlit web application for easy interaction
- 📄 **PDF Ingestion**: Automated processing of PDF technical manuals

## Architecture

The system follows a RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Ingestion** (`ingest.py`): PDF manuals are split into chunks and embedded using HuggingFace embeddings
2. **Vector Storage** (ChromaDB): Document chunks stored in vector database for efficient retrieval
3. **Query Processing** (`rag_core.py`): User questions are used to retrieve relevant context
4. **Answer Generation**: GPT-4 generates answers based on retrieved context
5. **Web Interface** (`app.py`): Streamlit UI for user interaction

## Project Structure

```
AutoTechQA/
├── src/
│   ├── app.py          # Streamlit web application
│   ├── config.py       # Configuration parameters
│   ├── ingest.py       # PDF ingestion and vector store creation
│   └── rag_core.py     # RAG chain implementation
├── data_source/        # Place PDF manuals here
├── vector_store/       # ChromaDB vector databases
├── requirements.txt    # Python dependencies
└── README.md
```

## Configuration

The system can be customized via `src/config.py`:

- **LLM Model**: `gpt-4` (default)
- **Embedding Model**: `nomic-ai/nomic-embed-text-v1`
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 100 characters
- **Top-K Retrieval**: 5 documents

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/indhra/AutoTechQA.git
   cd AutoTechQA
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### 1. Ingest PDF Manuals

Place your PDF manuals in the `data_source/` folder, then run:

```bash
python src/ingest.py
```

This will:
- Load PDFs from `data_source/`
- Process the first 30 pages of each manual
- Split text into chunks (1000 characters with 100 overlap)
- Create vector embeddings using HuggingFace nomic-embed model
- Store vectors in ChromaDB in `vector_store/`
- Skip already processed files by default

### 2. Run the Application

Start the Streamlit web interface:

```bash
streamlit run src/app.py
```

The application will be available at `http://localhost:8501`

### 3. Ask Questions

Enter your questions about trucks or buses in the web interface. The system will:
- Retrieve the 5 most relevant document chunks
- Use GPT-4 to generate an accurate answer
- Display the response in the UI

## Technical Stack

- **LangChain**: Framework for LLM applications and RAG pipelines
- **OpenAI GPT-4**: Large language model for answer generation
- **ChromaDB**: Vector database for similarity search
- **HuggingFace Transformers**: Embedding models (nomic-embed-text-v1)
- **Streamlit**: Web application framework
- **PyPDF**: PDF parsing and text extraction

## Dependencies

Key dependencies (see `requirements.txt` for complete list):
- streamlit
- langchain_core, langchain_openai, langchain_community
- langchain_huggingface, langchain-chroma
- chromadb
- sentence-transformers
- pypdf
- python-dotenv

## Notes

- The system currently processes only the first 30 pages of each PDF manual
- Existing vector stores are skipped during re-ingestion to save processing time
- All answers are strictly based on the provided context from manuals
- If information is not found in the manuals, the system explicitly states this

## Future Enhancements

Potential improvements:
- Support for more document formats
- Multi-language support
- Advanced filtering and metadata search
- Conversation history and context
- Source citation with page numbers
- Processing full PDFs instead of first 30 pages

## License

MIT License

## Contact
N A INDHRA KIRANU