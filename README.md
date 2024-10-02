# RAG with Groq, Pinecone, and Cohere

This project implements a Retrieval-Augmented Generation (RAG) system using Groq for language modeling, Pinecone for vector storage and retrieval, and Cohere for document reranking.

## Features

- PDF text extraction
- Text chunking and embedding
- Vector storage and retrieval with Pinecone
- Document reranking with Cohere
- Question answering using Groq's language model

## Technologies Used

- FastAPI
- Groq
- Pinecone
- Cohere
- PyPDF2
- LangChain

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/asheesh4545/rag-cohere-pinecone-groq.git
   cd rag-cohere-pinecone-groq
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   GROQ_API_KEY=your_groq_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```

## Usage

Run the FastAPI server:
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- POST `/query`: Send a question to get an answer based on the RAG system.

## Docker

A public Docker image is available:

```
docker pull asheesh4545/rag-cohere-pinecone-groq
```

## Deployment

The application is deployed on Render and accessible at:

[https://rag-cohere-pinecone-groq-latest.onrender.com/docs](https://rag-cohere-pinecone-groq-latest.onrender.com/docs)

Note: As the application is using the free tier, it may take some time to load initially.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
