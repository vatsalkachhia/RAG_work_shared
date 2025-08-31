# RAG Work Shared

A FastAPI-based RAG (Retrieval-Augmented Generation) application that allows users to upload documents and chat with them using AI-powered responses.

## Features

- **Session Management**: Create and manage user sessions with automatic cleanup
- **Document Upload**: Upload text files (.txt) or raw text to build knowledge bases
- **AI Chat**: Chat with uploaded documents using RAG technology
- **Multiple RAG Engines**: Support for various chunking, embedding, and vector database configurations
- **User Isolation**: Each user has their own isolated knowledge base and chat history

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd RAG_work_shared
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a requirements.txt file, install the required packages manually:
   ```bash
   pip install fastapi uvicorn pydantic python-multipart
   ```

3. **Install additional RAG dependencies** (if not already included)
   ```bash
   pip install langchain faiss-cpu sentence-transformers
   ```

## Running the Application

### Option 1: Using uvicorn directly

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Using Python

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Using the FastAPI CLI

```bash
fastapi run main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Create/Update Session
- **POST** `/session`
- **Headers**: `x-user-id` (optional)
- **Body**: `{"data": "session_data"}`

### 2. Upload Text
- **POST** `/upload-text`
- **Headers**: `x-user-id` (required)
- **Body**: `{"text": "your_text_content"}`

### 3. Upload File
- **POST** `/upload-file`
- **Headers**: `x-user-id` (required)
- **Body**: Form data with file (only .txt files supported)

### 4. Chat with Documents
- **POST** `/chat`
- **Headers**: `x-user-id` (required)
- **Body**: `{"question": "your_question"}`

## Usage Examples

### 1. Start a New Session
```bash
curl -X POST "http://localhost:8000/session" \
     -H "Content-Type: application/json" \
     -d '{"data": "initial_session_data"}'
```

### 2. Upload Text Content
```bash
curl -X POST "http://localhost:8000/upload-text" \
     -H "Content-Type: application/json" \
     -H "x-user-id: your_user_id" \
     -d '{"text": "Your document content here..."}'
```

### 3. Upload a Text File
```bash
curl -X POST "http://localhost:8000/upload-file" \
     -H "x-user-id: your_user_id" \
     -F "file=@your_document.txt"
```

### 4. Chat with Your Documents
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -H "x-user-id: your_user_id" \
     -d '{"question": "What is this document about?"}'
```

## Configuration

The application uses default RAG configuration that can be modified in `routes.py`:

```python
DEFAULT_RAG_CONFIG = {
    "chunking": "recursive",
    "embedding": "huggingface",
    "vectordb": "faiss",
    "retrieval": "topk",
    "llm": "groq",
    "memory": "windowed",
    "reranker": False,
}
```

## Session Management

- Sessions automatically expire after 1 hour of inactivity
- Cleanup runs every 5 minutes
- Each user gets their own isolated RAG engine and knowledge base

## Development

### Project Structure
```
RAG_work_shared/
├── main.py          # FastAPI application entry point
├── routes.py        # API route definitions and handlers
├── engine.py        # RAG engine implementation
└── README.md        # This file
```

### Adding New Routes

To add new routes, simply add them to `routes.py` using the `@router` decorator:

```python
@router.get("/new-endpoint")
async def new_endpoint():
    return {"message": "New endpoint"}
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port number in the uvicorn command
2. **Import errors**: Ensure all dependencies are installed
3. **File upload errors**: Make sure files are .txt format and x-user-id header is provided

### Logs

The application will show startup logs and any errors in the console. For production, consider adding proper logging configuration.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions, please create an issue in the repository or contact the maintainers.
