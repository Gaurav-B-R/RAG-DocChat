# RAG Chatbot Application

## Overview
This application is a Retrieval-Augmented Generation (RAG) chatbot that processes documents (PDF or HTML) and enables users to query the content in natural language. Additionally, it supports processing web pages via URLs.

### How This RAG Application is Built
This RAG chatbot is designed to enhance information retrieval using a hybrid approach that combines **semantic search** and **keyword-based search**. It leverages:
- **FastAPI** for building an efficient and scalable backend
- **Sentence Transformers** for generating high-quality embeddings to perform similarity-based retrieval
- **Scikit-learn** for implementing cosine similarity for relevance ranking
- **BeautifulSoup** and **PyPDF2** for processing HTML and PDF documents
- **Google Gemini 2.0 Pro API (LLM)** for advanced natural language responses
- **Uvicorn** as an ASGI server to handle requests asynchronously

### Advanced Engineering Mechanisms
This application incorporates several engineering mechanisms:
- **Efficient Chunking**: Documents are split into context-aware chunks, ensuring optimal retrieval of relevant information.
- **Memory Persistence**: The system saves processed documents and their embeddings to disk to enhance session continuity and prevent redundant processing.
- **Hybrid Search Strategy**: A combination of **semantic similarity** and **keyword matching** is used to retrieve the most relevant information.
- **CORS Handling**: The API is designed with Cross-Origin Resource Sharing (CORS) policies, allowing seamless interactions with frontend applications.

### How It Functions
1. **Document Ingestion**: The user uploads a document (PDF/HTML) or provides a webpage URL.
2. **Text Processing & Chunking**: The document is parsed, cleaned, and split into meaningful segments.
3. **Embedding Generation**: Each chunk is converted into an embedding using **Sentence Transformers**.
4. **Retrieval & Querying**: When a user submits a query, relevant document chunks are retrieved using **cosine similarity** and keyword matching.
5. **Augmented Response Generation**: The retrieved information is fed into **Gemini 2.0 Pro API (LLM)**, which generates a coherent and enriched response.
6. **Frontend Interaction**: Users can interact with the chatbot via a user-friendly **chat interface (chat.html)** or use API endpoints for programmatic access.

## Prerequisites
Ensure you have the following installed on your system:
- Python 3.8+
- `pip` (Python package manager)
- `git` (optional, for cloning the repository)

## Skip Local Setup - Use Hosted Version
To skip the local setup and installation, I have made it even simpler by hosting this application on **Render**. You can visit the following URL to use the application instantly:

🔗 **[RAG Chatbot on Render](https://rag-qe3s.onrender.com/)**

Simply open the link in your browser, upload documents, and start querying them instantly!

## Setup Instructions

### 1. Clone the Repository
```sh
$ git clone <repository-url>
$ cd <repository-folder>
```

### 2. Create a Virtual Environment (Recommended)

**Note:** Select and use Python (preferably **3.11**) but ensure that the version is **less than 3.13**, as many libraries aren't fully compatible with Python 3.13 yet.
```sh
$ python -m venv venv
$ source venv/bin/activate  # On macOS/Linux
$ venv\Scripts\activate    # On Windows
```

### 3. Upgrade Pip, Setuptools, and Wheel
Before installing dependencies, upgrade `pip`, `setuptools`, and `wheel` to ensure compatibility and avoid potential issues:
```sh
$ pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies
```sh
$ pip install -r requirements.txt
```

### 4. Set Up API Keys
This application uses Gemini 2.0 Pro API (LLM). You need to obtain an API key by:
1. Visiting [AI Studio](https://aistudio.google.com/)
2. Clicking the **Get API Key** option from the top left corner
3. Selecting **Create API Key** to generate your key
4. Adding the key to `app.py` where required

#### Where to Add the API Key
For testing purposes, we are directly using the API key inside `app.py`:
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY_HERE")
```
However, in a production environment, it is recommended to store the API key securely using environment variables or a `.env` file:

### 5. Run the Application
```sh
$ uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
The application will start running at `http://localhost:8000`.

## Testing the Application

### 1. Primary Testing Method - Web Interface (Easy and Simple)
To make testing easier, a user-friendly **chat interface** has been created using `chat.html`. Once the application is running, visit:
```sh
http://0.0.0.0:8000/
```
This will load the chat interface, where you can upload documents, process URLs, and ask questions comfortably.

Additionally, for API testing and interacting with all available endpoints, you can visit the automatically generated API documentation at:
```sh
http://0.0.0.0:8000/docs
```
This provides an interactive Swagger UI where you can test the application's functionalities directly by sending requests to various endpoints.

### 2. Upload a Document via API
You can upload a document (PDF/HTML) using the web interface at `http://0.0.0.0:8000` or via API:
```sh
$ curl -X 'POST' \
  'http://localhost:8000/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your-document.pdf'
```

### 3. Process a Web URL via API
You can also process a webpage by providing a URL:
```sh
$ curl -X 'POST' \
  'http://localhost:8000/upload_url' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com"}'
```

### 4. Query the Chatbot via API
Once a document or URL is processed, you can query it via the web interface or API:
```sh
$ curl -X 'GET' \
  'http://localhost:8000/query?q=What%20is%20this%20document%20about?' \
  -H 'accept: application/json'
```

### 5. Reset the Application State
To reset and remove all uploaded documents:
```sh
$ curl -X 'POST' 'http://localhost:8000/reset'
```
```sh
$ uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
The application will start running at `http://localhost:8000`.

## Additional Documentation

For detailed documentation and in-depth information about the RAG application, please refer to the documentation available in the Git repository.

This project is open-source and available under the MIT License.

## Contact
For issues or contributions, submit a pull request or open an issue in the repository.

**Name:** Gaurav Bharatavalli Rangaswamy  
**GitHub:** [Gaurav-B-R](https://github.com/Gaurav-B-R)  
**Email:** gauravhsn8@gmail.com

