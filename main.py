import os
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader as LangChainPyPDFLoader
from langchain_chroma import Chroma
from groq import Groq
import chromadb

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace '*' with specific domains for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
db = None  # Global database variable
persist_directory = 'VectorDB_storage/chromadb'

# Define the input model for the query endpoint
class QueryRequest(BaseModel):
    user_query: str

# Function to add documents to the database
def add_to_db(file_path: str):
    global db
    # Load and split PDF document
    loader = LangChainPyPDFLoader(file_path)
    chunks = loader.load_and_split()

    # Set up embeddings and database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Initialize Chroma database with all chunks and embeddings
    if db is None:
        db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    else:
        db.add_documents(chunks)

    print("Database created and populated with embeddings for all documents.")

def load_initialize_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    persistent_client = chromadb.PersistentClient(path=persist_directory)
    vector_store = Chroma(
                client=persistent_client,
                # collection_name=collection_name,
                embedding_function=embeddings,
            )

@app.post("/add-documents")
async def add_documents(file: UploadFile):
    try:
        # Ensure the temp directory exists
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Add the document to the database
        add_to_db(file_path)

        # Cleanup temporary file
        os.remove(file_path)

        return {"message": f"Document '{file.filename}' added to the database successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def answer_query(request: QueryRequest):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    persistent_client = chromadb.PersistentClient(path=persist_directory)
    db = Chroma(
                client=persistent_client,
                # collection_name=collection_name,
                embedding_function=embeddings,
            )
    user_query = request.user_query
    context = db.similarity_search(user_query)
    # Check if relevant context is found
    if not context:
        raise HTTPException(status_code=404, detail="No relevant content found in the provided context.")
    # Get response from OpenAI API
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained on user provided pdfs."},
            {"role": "user", "content": f"""
            You are a helpful assistant. Use the following provided context to answer the user's query.
            Context: {context}
            User Query: {user_query}
            """}
        ]
    )
    llm_response = response.choices[0].message.content.strip()
    return {"response": llm_response}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Document Query API!"}

# Run the app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=11000)