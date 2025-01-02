# Retrieval-Augmented Generation (RAG) API for Document Querying

This open-source project provides an API for uploading PDF documents, generating vector embeddings, storing them in a vector database, and answering user queries using Retrieval-Augmented Generation (RAG). The API combines the power of document retrieval with AI-based natural language understanding to deliver accurate and context-aware responses.

## Features

- **Upload PDF Documents**: Efficiently process and store documents for query-based retrieval.
- **Vector Embedding Storage**: Uses ChromaDB for storing vector embeddings and retrieval.
- **Retrieval-Augmented Generation**: Combines document retrieval with Groq's AI model to provide rich, contextually aware answers.
- **FastAPI Framework**: Built on FastAPI for high performance and scalability.
- **CORS Enabled**: Supports cross-origin requests, enabling frontend integrations.

## Prerequisites

- Python 3.8 or higher
- [Groq API Key](https://groq.com)
- [ChromaDB](https://docs.trychroma.com)
- [FastAPI](https://fastapi.tiangolo.com/)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/rag-query-api.git
   cd rag-query-api
2. **Set up a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt

```

4. **Configure environment variables**:
Create a .env file in the root directory.
Add your Groq API Key:
.env file
```bash
GROQ_API_KEY=your_groq_api_key
```

***Run the application***:

```bash
uvicorn main:app --host 127.0.0.1 --port 11000
```
API Endpoints
1. **Upload Documents**
Endpoint: /upload-documents
Method: POST
Description: Upload a PDF document to generate embeddings and store in the vector database.

Request: Form-data with a file field.

Response:
json
```bash
{
  "message": "Document 'example.pdf' added to the database successfully."
}
```
2. **Ask a Question**
Endpoint: /ask_question
Method: POST
Description: Submit a query to retrieve relevant context from the document and generate a response.

Request:
json
```bash
{
  "user_query": "What does this document say about topic X?"
}
```
Response:
json
```bash
{
  "response": "Here is the answer based on the context of your documents..."
}
```
