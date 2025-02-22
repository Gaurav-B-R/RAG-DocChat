import io
import os
import asyncio
import numpy as np
import uvicorn
import httpx
from urllib.parse import urlparse
from typing import Union
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time
from pathlib import Path

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

genai.configure(api_key="YOUR_API_KEY_HERE") 

# Load a pre-trained embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global in-memory storage for document chunks.
# Each entry is a dict with keys "text" and "embedding"
document_chunks = []

# Global in-memory storage for document metadata
document_metadata = {
    "title": None,
    "type": None,
    "source": None
}

# Add session tracking
last_activity = None
session_timeout = 3600  # 1 hour in seconds

# Add storage configuration
STORAGE_DIR = Path(__file__).parent / "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)  # Create directory if it doesn't exist

def save_document_state():
    """Save the current document state to disk."""
    try:
        state = {
            "chunks": document_chunks,
            "metadata": document_metadata,
            "timestamp": time.time()
        }
        with open(STORAGE_DIR / "document_state.pkl", "wb") as f:
            pickle.dump(state, f)
    except Exception as e:
        print(f"Error saving document state: {e}")

def load_document_state() -> bool:
    """Load document state from disk."""
    global document_chunks, document_metadata
    try:
        state_file = STORAGE_DIR / "document_state.pkl"
        if not state_file.exists():
            return False
            
        # Check if state is too old (e.g., more than 24 hours)
        if time.time() - state_file.stat().st_mtime > 86400:
            return False
            
        with open(state_file, "rb") as f:
            state = pickle.load(f)
            
        document_chunks = state["chunks"]
        document_metadata = state["metadata"]
        return True
    except Exception as e:
        print(f"Error loading document state: {e}")
        return False

def verify_document_state() -> bool:
    """Verify if the document is loaded and valid."""
    global document_chunks, document_metadata
    
    # If no document in memory, try loading from disk
    if not document_chunks:
        if not load_document_state():
            return False
    
    try:
        # Verify chunks and metadata
        if not document_chunks or len(document_chunks) == 0:
            return False
            
        for chunk in document_chunks:
            if not isinstance(chunk, dict) or 'text' not in chunk or 'embedding' not in chunk:
                return False
                
        if not document_metadata or not document_metadata.get('title'):
            return False
            
        return True
    except Exception:
        return False

def clear_document_state():
    """Clear all document state including stored files."""
    global document_chunks, document_metadata
    
    # Clear memory
    document_chunks = []
    document_metadata = {
        "title": None,
        "type": None,
        "source": None
    }
    
    # Clear stored files
    try:
        for file in STORAGE_DIR.glob("*.pkl"):
            file.unlink()
    except Exception as e:
        print(f"Error clearing stored files: {e}")

### Helper functions for document processing and indexing ###

def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def parse_html(file_bytes: bytes) -> dict:
    """Extract text and metadata from an HTML file."""
    soup = BeautifulSoup(file_bytes, "html.parser")
    
    # Extract title
    title = soup.title.string if soup.title else "Untitled HTML Document"
    
    # Only remove script and style tags, keep other structural elements
    for element in soup(['script', 'style']):
        element.decompose()
    
    # Collect text from all important sections
    important_sections = []
    
    # Look for pricing related sections first
    pricing_sections = soup.find_all(['div', 'section', 'article'], 
                                   class_=lambda x: x and any(term in x.lower() 
                                   for term in ['price', 'pricing', 'plan', 'subscription', 'cost']))
    
    for section in pricing_sections:
        important_sections.append(section.get_text(separator=' ', strip=True))
    
    # Get main content
    main_content = soup.find('main') or soup.find('article') or soup.find('body')
    if main_content:
        # Process all paragraphs and sections
        for element in main_content.find_all(['p', 'div', 'section', 'article', 'table']):
            text = element.get_text(separator=' ', strip=True)
            if text and len(text) > 20:  # Only keep substantial content
                important_sections.append(text)
    
    # Extract meta description
    meta_desc = ""
    meta_tags = soup.find_all('meta', attrs={'name': ['description', 'keywords'], 
                                            'property': ['og:description', 'og:title']})
    for tag in meta_tags:
        meta_desc += tag.get('content', '') + " "
    
    # Combine all content with clear section breaks
    content = "\n\n".join(important_sections)
    
    return {
        "title": title,
        "content": content,
        "description": meta_desc.strip()
    }

def chunk_text(text: str, max_words: int = 150, overlap: int = 30) -> list:
    """Break text into overlapping chunks to preserve context."""
    # First split by double newlines to preserve natural breaks
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue
            
        # If a single paragraph is too long, split it
        if len(words) > max_words:
            for i in range(0, len(words), max_words - overlap):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk)
        else:
            # Try to combine shorter paragraphs
            if current_length + len(words) <= max_words:
                current_chunk.extend(words)
                current_length += len(words)
            else:
                # Store current chunk and start a new one
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = words
                current_length = len(words)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

### Dummy Embedding & Similarity Functions ###

def dummy_embedding(text: str) -> np.ndarray:
    """
    Create a dummy embedding by counting normalized letter frequencies.
    """
    vector = np.zeros(26)
    for char in text.lower():
        if 'a' <= char <= 'z':  # Ensure the character is a lowercase letter
            index = ord(char) - ord('a')
            vector[index] += 1
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

def get_embedding(text: str):
    """Generate an embedding for the given text using a transformer model."""
    return embedding_model.encode(text)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 > 0 and norm2 > 0:
        return dot / (norm1 * norm2)
    return 0.0

### Search Functions ###

def keyword_search(query: str) -> list:
    """A simple keyword search that returns chunks containing the query text."""
    results = []
    query_lower = query.lower()
    for chunk in document_chunks:
        if query_lower in chunk["text"].lower():
            results.append(chunk["text"])
    return results

def semantic_search(query: str, top_k: int = 7) -> list:
    """Perform semantic search with improved relevance scoring."""
    query_emb = get_embedding(query)
    scored_chunks = []
    
    # Extract key terms from query for additional matching
    query_terms = set(query.lower().split())
    
    for chunk in document_chunks:
        # Calculate semantic similarity
        sim = cosine_similarity(query_emb, chunk["embedding"])
        
        # Calculate term overlap score
        chunk_terms = set(chunk["text"].lower().split())
        term_overlap = len(query_terms.intersection(chunk_terms)) / len(query_terms) if query_terms else 0
        
        # Combined score with weighted components
        combined_score = (sim * 0.7) + (term_overlap * 0.3)
        
        scored_chunks.append({
            "text": chunk["text"],
            "score": combined_score,
            "semantic_sim": sim,
            "term_overlap": term_overlap
        })

    # Sort by combined score and filter
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Dynamic threshold based on highest score
    if scored_chunks:
        max_score = scored_chunks[0]["score"]
        threshold = max(0.05, max_score * 0.3)  # At least 30% as relevant as best match
        
        return [chunk["text"] for chunk in scored_chunks[:top_k] 
                if chunk["score"] > threshold]
    return []

def combine_results(keyword_results: list, semantic_results: list) -> str:
    """Combine results from both searches and remove duplicates."""
    combined = list(set(keyword_results + semantic_results))
    # Join chunks with a clear delimiter
    return "\n---\n".join(combined)

def analyze_query_type(query: str) -> dict:
    """Analyze the query to determine its type and requirements."""
    query_lower = query.strip().lower()
    
    # Query type analysis
    analysis = {
        "is_greeting": is_greeting(query_lower),
        "is_comparison": any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better', 'pros and cons']),
        "is_listing": any(word in query_lower for word in ['list', 'what are', 'tell me all', 'enumerate', 'show me', 'give me']),
        "is_definition": any(word in query_lower for word in ['what is', 'define', 'explain', 'describe', 'meaning of']),
        "is_summary": any(word in query_lower for word in ['summarize', 'summary', 'brief', 'overview', 'tldr']),
        "is_example": any(word in query_lower for word in ['example', 'instance', 'show me an example', 'sample']),
        "is_how_to": any(word in query_lower for word in ['how to', 'how do i', 'steps to', 'guide', 'procedure']),
        "is_why": query_lower.startswith('why') or 'reason for' in query_lower or 'explain why' in query_lower,
        "requires_context": not (is_greeting(query_lower) or query_lower.startswith('hi') or query_lower.startswith('hello'))
    }
    
    # Extract key terms for focused search
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
    key_terms = [word for word in query_lower.split() if word not in stop_words]
    analysis["key_terms"] = key_terms
    
    return analysis

def get_response_format(analysis: dict) -> str:
    """Get appropriate response format based on query analysis."""
    if analysis["is_comparison"]:
        return """
        Format your response as a clear comparison:
        1. First, identify the items being compared
        2. Use a structured format with clear headings
        3. Present key differences and similarities
        4. Include a brief summary of the comparison
        """
    elif analysis["is_listing"]:
        return """
        Format your response as a clear list:
        • Use bullet points for better readability
        • Group related items together
        • Provide brief explanations where needed
        • Keep the formatting consistent
        """
    elif analysis["is_how_to"]:
        return """
        Format your response as a step-by-step guide:
        1. Start with any prerequisites
        2. Present steps in a logical order
        3. Include any important cautions or notes
        4. End with expected outcomes
        """
    elif analysis["is_summary"]:
        return """
        Format your response as a concise summary:
        • Start with the main point
        • Include only key information
        • Use clear, direct language
        • End with any important conclusions
        """
    else:
        return """
        Format your response clearly:
        • Provide a direct answer first
        • Include supporting details
        • Use examples where helpful
        • Maintain clarity and precision
        """

def build_prompt(query: str) -> str:
    """Construct an improved prompt for more precise answers."""
    # Analyze the query
    analysis = analyze_query_type(query)
    
    # Handle greetings without accessing the document
    if analysis["is_greeting"]:
        return query

    if not document_chunks:
        return "No document has been uploaded yet. Please upload a document or provide a URL."

    # Handle metadata queries
    if any(word in query.lower() for word in ['title', 'called', 'name of']):
        title = document_metadata.get('title', 'Untitled')
        source_type = document_metadata.get('type', 'document')
        source = document_metadata.get('source', '')
        if source_type == 'URL':
            return f"""METADATA_RESPONSE: The webpage titled "{title}" is from {source}"""
        return f"""METADATA_RESPONSE: The title of the {source_type} is "{title}"."""

    # Get relevant document chunks
    kw_results = keyword_search(query)
    sem_results = semantic_search(query)
    retrieved_context = combine_results(kw_results, sem_results)

    if not retrieved_context:
        return f"Question: {query}\nAnswer: I couldn't find specific information about '{query}' in the document. Could you rephrase your question or ask about a different aspect?"

    # Get document metadata
    doc_type = document_metadata.get('type', 'document')
    doc_title = document_metadata.get('title', 'Untitled')
    doc_desc = document_metadata.get('description', '')

    # Get appropriate response format
    response_format = get_response_format(analysis)

    # Build the enhanced prompt
    prompt = f"""You are a precise and thorough analysis assistant working with a {doc_type} titled "{doc_title}".
    {f'Document description: {doc_desc}' if doc_desc else ''}

    User Question: {query}
    
    Query Analysis:
    - Type: {', '.join(k for k, v in analysis.items() if v and k.startswith('is_') and k != 'is_greeting')}
    - Key Terms: {', '.join(analysis['key_terms'])}
    
    Context Excerpts:
    {retrieved_context}
    
    Instructions:
    1. Focus on answering the specific question asked
    2. Use only information from the provided excerpts
    3. If information is incomplete or unclear, state this explicitly
    4. Cite specific details and examples from the text
    5. Maintain technical accuracy and precision
    6. If multiple interpretations are possible, explain the alternatives
    
    {response_format}
    
    Answer:"""
    
    return prompt

### Simulated Gemini 2.0 Flash Integration ###

async def real_gemini_response(prompt: str):
    """Calls Google Gemini API to generate a structured response."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        
        if hasattr(response, "text"):  # Ensure response has text
            return response.text
        return "Error: No response text received from Gemini."
    
    except Exception as e:
        return f"Error: {str(e)}"

def format_response(text: str) -> str:
    """Format the response for better readability."""
    if not text.strip():
        return text
        
    # Handle table formatting
    if '|' in text and '-|-' in text:
        lines = text.split('\n')
        formatted_lines = []
        header_processed = False
        
        for line in lines:
            if not line.strip():
                continue
                
            # Process table header
            if '|' in line and not header_processed:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                formatted_lines.append("\n" + "=" * 80)  # Table top border
                formatted_lines.append(f"  {cells[0]:<20} {cells[1]}")  # Header
                formatted_lines.append("=" * 80)  # Header separator
                header_processed = True
                continue
                
            # Skip separator line
            if '-|-' in line:
                continue
                
            # Process table rows
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(cells) >= 2:
                    formatted_lines.append(f"  {cells[0]:<20} {cells[1]}")
                    
        formatted_lines.append("=" * 80 + "\n")  # Table bottom border
        return '\n'.join(formatted_lines)
    
    # Handle bullet points and lists
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                formatted_lines.append("")  # Add space between list items
            continue
            
        # Handle bullet points
        if line.startswith('•') or line.startswith('-'):
            in_list = True
            formatted_lines.append(f"  {line}")  # Indent bullet points
        # Handle numbered lists
        elif line[0].isdigit() and line[1:3] in ['. ', ') ']:
            in_list = True
            formatted_lines.append(f"  {line}")  # Indent numbered lists
        else:
            in_list = False
            formatted_lines.append(line)
            
    return '\n'.join(formatted_lines)

def is_greeting(text: str) -> bool:
    """Check if the input is a greeting."""
    greetings = {
        'hi', 'hello', 'hey', 'howdy', 'hola', 'greetings', 
        'good morning', 'good afternoon', 'good evening',
        'hi there', 'hello there'
    }
    return text.lower().strip() in greetings

def get_greeting_response() -> str:
    """Return a friendly greeting response."""
    return """Hello! I'm ready to help you understand the document. You can ask me specific questions about its content, and I'll do my best to provide accurate answers. What would you like to know?"""

async def stream_gemini_response(prompt: str):
    """Streams response from Gemini API with improved formatting."""
    try:
        # Verify document state first
        if not verify_document_state() and not is_greeting(prompt):
            yield "I apologize, but I've lost connection to the document. Please try uploading it again."
            return
            
        # Handle greetings
        if is_greeting(prompt):
            yield get_greeting_response()
            return
            
        # Handle metadata responses
        if prompt.startswith("METADATA_RESPONSE:"):
            yield prompt.replace("METADATA_RESPONSE:", "").strip() + "\n"
            return
            
        model = genai.GenerativeModel("gemini-pro")
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "block_none",
            "HARM_CATEGORY_HATE_SPEECH": "block_none",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
        }
        
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Get the response without streaming first
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False
        )
        
        if not hasattr(response, "text"):
            yield "I apologize, but I couldn't generate a response at the moment. Please try asking your question again."
            return
            
        # Format the complete response
        formatted_response = format_response(response.text)
        
        # Stream the formatted response line by line
        for line in formatted_response.split('\n'):
            if line.strip():  # Only yield non-empty lines
                yield line + '\n'
                await asyncio.sleep(0.01)
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            yield "I apologize, but I'm experiencing high traffic at the moment. Please try again in a few seconds."
        else:
            yield "I encountered an error processing your request. Please try uploading the document again or rephrase your question."

### API Endpoints ###

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF or HTML), extract and chunk text, and index it."""
    try:
        # Validate file type
        if file.content_type not in ["application/pdf", "text/html"]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {file.content_type}. Please upload PDF or HTML files only."
            )
        
        # Clear existing document state
        clear_document_state()
        
        # Read file content with size limit
        content = bytearray()
        size = 0
        chunk_size = 8192  # 8KB chunks
        
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            if size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=413, detail="File too large (max 10MB)")
            content.extend(chunk)

        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        # Process the document
        try:
            if file.content_type == "application/pdf":
                parsed_doc = parse_pdf(content)
                document_metadata = {
                    "title": parsed_doc["title"] if isinstance(parsed_doc, dict) else file.filename,
                    "type": "PDF",
                    "source": f"uploaded_file: {file.filename}",
                    "description": parsed_doc.get("description", "") if isinstance(parsed_doc, dict) else ""
                }
                text = parsed_doc["content"] if isinstance(parsed_doc, dict) else parsed_doc
            elif file.content_type in ["text/html", "application/octet-stream"]:
                parsed_html = parse_html(content)
                text = parsed_html["content"]
                document_metadata = {
                    "title": parsed_html["title"],
                    "type": "HTML",
                    "source": f"uploaded_file: {file.filename}",
                    "description": parsed_html["description"]
                }
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            
            chunks = chunk_text(text)
            for chunk in chunks:
                embedding = get_embedding(chunk)
                document_chunks.append({"text": chunk, "embedding": embedding})
                
            # Ensure storage directory exists
            os.makedirs(STORAGE_DIR, exist_ok=True)
            
            # Save state after successful processing
            save_document_state()
            
            return {
                "message": f"Document '{file.filename}' processed successfully",
                "num_chunks": len(document_chunks),
                "document_info": document_metadata
            }
            
        except Exception as e:
            print(f"Document processing error: {str(e)}")  # Add logging
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload_url")
async def upload_url(url: str):
    """Upload and process a webpage from URL."""
    # Clear existing document state first
    clear_document_state()
    
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    try:
        content = await fetch_url_content(url)
        global document_metadata
        global document_chunks
        
        document_chunks = []
        
        # Parse the HTML content
        parsed_html = parse_html(content)
        text = parsed_html["content"]
        document_metadata = {
            "title": parsed_html["title"],
            "type": "URL",
            "source": url,
            "description": parsed_html["description"]
        }
        
        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = get_embedding(chunk)
            document_chunks.append({"text": chunk, "embedding": embedding})
            
        # Save state after successful processing
        save_document_state()
        
        return {
            "message": "URL processed and indexed successfully",
            "num_chunks": len(document_chunks),
            "document_info": document_metadata
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

def is_valid_url(url: str) -> bool:
    """Check if the URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

async def fetch_url_content(url: str) -> bytes:
    """Fetch content from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

@app.get("/status")
async def get_status():
    """Check if a document is loaded and get basic stats."""
    is_loaded = len(document_chunks) > 0 and document_metadata.get("title") is not None
    return {
        "document_loaded": is_loaded,
        "num_chunks": len(document_chunks) if is_loaded else 0,
        "document_info": document_metadata if is_loaded else None,
        "sample_chunk": document_chunks[0]["text"][:200] if is_loaded else None
    }

@app.get("/stream_query")
async def stream_query(q: str):
    """Process a query with document state verification."""
    if not verify_document_state():
        return JSONResponse({
            "error": "Document state invalid",
            "message": "Please upload your document again to continue the conversation."
        })

    prompt = build_prompt(q)
    return StreamingResponse(stream_gemini_response(prompt), media_type="text/plain")

@app.get("/query")
async def query(q: str):
    """Process a query with document state verification."""
    if not verify_document_state():
        return JSONResponse({
            "error": "Document state invalid",
            "message": "Please upload your document again to continue the conversation."
        })
        
    prompt = build_prompt(q)
    response_text = await real_gemini_response(prompt)
    return JSONResponse({"answer": response_text})

@app.post("/reset")
async def reset_state():
    """Reset the application state completely."""
    clear_document_state()
    return {"message": "Application state reset successfully"}

@app.post("/cleanup")
async def cleanup_old_files():
    """Remove old document states."""
    try:
        for file in STORAGE_DIR.glob("*.pkl"):
            if time.time() - file.stat().st_mtime > 86400:  # 24 hours
                file.unlink()
        return {"message": "Old files cleaned up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

# Mount the static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Add a root endpoint to serve the chat interface
@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "chat.html"))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
