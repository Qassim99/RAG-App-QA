from fastapi import FastAPI, APIRouter, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import aiofiles
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

# ollama 
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings


import dotenv
dotenv.load_dotenv()

# get gemini api key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# set gemini api key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

gemini_model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.0,
                    max_tokens=1024
                )

llava_model = ChatOllama(
    model="llava:latest",
    temperature=0.0,
    num_predict=1024
)

phi4_model = ChatOllama(
    model="phi4:latest",
    temperature=0.0,
    num_predict=1024
)




app = FastAPI()

base_router = APIRouter()

@base_router.get("/")
async def root():
    return {"message": "Hello World from base"}




@base_router.post("/upload/{project_id}")
async def upload(project_id: str, file: UploadFile):
   
    if file.content_type not in ["text/plain", "application/pdf", "image/png", "image/jpeg"] or file.size > 10000000:
        return JSONResponse(status_code=400, content={"message": "Invalid file type or size"})

    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_dir = os.path.join(base_dir, "files", project_id)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, file.filename)

    # use chunk to save file instead to save the entire file, which not efficiancy for momery 
    async with aiofiles.open(file_path, "wb") as f:
        while chunk := await file.read(1024):
            await f.write(chunk)

    return JSONResponse(status_code=200, content={"message": "File uploaded successfully"})



class ProcessRequest(BaseModel):
    # request the chunk size and overlay 
    chunk_size: int
    overlap: int


@base_router.post("/process/{project_id}")
async def process(project_id: str, request: ProcessRequest):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_dir = os.path.join(base_dir, "files", project_id)
    if not os.path.exists(file_dir):
        return JSONResponse(status_code=400, content={"message": "Project not found"})
    
    # read all file in project directory and extract text and image from pdf
    # and and turn into chunks
    
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=request.chunk_size, chunk_overlap=request.overlap)
    for filename in os.listdir(file_dir):
        file_path = os.path.join(file_dir, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            # split text into chunks and add metadata
            file_content_texts = [
                doc.page_content
                for doc in documents
            ]

            file_content_metadatas = [
                doc.metadata
                for doc in documents
            ]
            subChunks = text_splitter.create_documents(
                file_content_texts,
                metadatas=file_content_metadatas
            )
            chunks.extend(subChunks)

        elif filename.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path,
                                   extract_images=True,
                                   images_parser=LLMImageBlobParser(
                                        model= llava_model
                                   ),
                                   images_inner_format="markdown-img"
                                   )
            documents = loader.load()
            # split text into chunks and add metadata
            file_content_texts = [
                doc.page_content
                for doc in documents
            ]
            file_content_metadatas = [
                doc.metadata
                for doc in documents
            ]
            subChunks = text_splitter.create_documents(
                file_content_texts,
                metadatas=file_content_metadatas
            )
            chunks.extend(subChunks)
            
        elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            loader = LLMImageBlobParser(
                model=llava_model
            )
            # Create Blob from file path (not raw bytes)
            blob = Blob.from_path(file_path)
            documents = loader.parse(blob)

            for document in documents:
                # split text into chunks and add metadata
                file_content_texts = [document.page_content]
                file_content_metadatas = [document.metadata]
                subChunks = text_splitter.create_documents(
                    file_content_texts,
                    metadatas=file_content_metadatas
                )
                chunks.extend(subChunks)
            
        else:
            continue

    if not chunks or len(chunks) == 0:
        return JSONResponse(status_code=400, content={"message": "No valid documents found"})
    else:
        try:
            # create embedddings for chunks and store in chroma vector db
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
            # use in memory byte store for chroma

            # batch_size = 10  # You can adjust this batch size as needed

            # # Prepare texts and metadatas for batching
            # texts = [chunk.page_content for chunk in chunks]
            # metadatas = [chunk.metadata for chunk in chunks]
            # all_embeddings = []
            # for text_batch in batch_iterable(texts, batch_size):
            #     print(f"Processing batch of size: {text_batch}")
            #     print("="*100)
            #     batch_embeds = embeddings.embed_documents(text_batch)
            #     all_embeddings.extend(batch_embeds)

            
            # vector_store = Chroma.from_embeddings(
            #     embeddings=all_embeddings,
            #     texts=texts,
            #     metadatas=metadatas,
            #     collection_name=f"project_{project_id}",
            #     persist_directory="./chroma_db"
            # )

            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=f"project_{project_id}",
                persist_directory="./chroma_db"
            )

            store = InMemoryByteStore()
            

            retriever = MultiVectorRetriever(
                vectorstore=vector_store,
                docstore=store,
                search_kwargs={"k": 5}
            )
        except Exception as e:
            return JSONResponse(
                status_code=429,
                content={"message": f"Embedding API error: {str(e)}"}
            )

        # Serialize chunks for JSON response
        serialized_chunks = [
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        return JSONResponse(status_code=200, content={"message": serialized_chunks})



class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@base_router.post("/query/{project_id}")
async def query(project_id: str, request: QueryRequest):
    try:
        
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        vector_store = Chroma(
            collection_name=f"project_{project_id}",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Retrieve relevant documents
        retriever = vector_store.as_retriever(
            search_kwargs={"k": request.top_k}
        )
        docs = retriever.invoke(request.query)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt with context
        prompt = f"""Based on the following context, answer the question and make sure to refine the answer.
        
        Context: {context}

        Question: {request.query}

        Answer:"""
                
        # Generate response using gemini model
        response = gemini_model.invoke(prompt)
        
        return JSONResponse(status_code=200, content={
            "answer": response.content,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ]
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Query error: {str(e)}"}
        )

# def batch_iterable(iterable, batch_size):
#     """Yield successive batches from iterable."""
#     for i in range(0, len(iterable), batch_size):
#         yield iterable[i:i + batch_size]









