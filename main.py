import os
from typing import Dict

import PyPDF2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
# from langchain_groq import ChatGroq
import pinecone
import cohere
from groq import Groq

app = FastAPI()

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Initialize clients
# llm_groq = ChatGroq(temperature=0, model_name="groq/mixtral-8x7b-32768", api_key=GROQ_API_KEY)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
co = cohere.Client(COHERE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize FastEmbedEmbeddings
embeddings = FastEmbedEmbeddings()

# Create or get Pinecone index
index = pc.Index("testv2")

def read_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    return " ".join(page.extract_text() for page in pdf.pages)

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_query_embedding(text):
    embeddings = FastEmbedEmbeddings()
    return embeddings.embed_query(text)

def query_index(query_embedding, top_k=5):
    return index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

def rerank_documents(query, documents, top_n=5):
    return co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=True
    )

# def generate_response(context, query):
#     template = f"Based on the following context: {context} generate precise summary related to question: {query} Do not remove necessary information related to context. Consider `\\n` as newline character."
    
#     chat_completion = groq_client.chat.completions.create(
#         messages=[{"role": "user", "content": template}],
#         model="mixtral-8x7b-32768",
#     )
    
#     return chat_completion.choices[0].message.content

def generate_response(context, query):
    template = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question accurately, state that the available information is insufficient to provide a reliable answer.

Context: {context}

Question: {query}

Instructions:
1. If the context is relevant and sufficient:
   - Provide a precise and informative answer.
   - Include all necessary details from the context.
   - Use `\n` for line breaks to improve readability.

2. If the context is insufficient or irrelevant:
   - Clearly state that the available information is not enough to answer the question accurately.
   - Do not attempt to guess or provide information not supported by the given context.

Answer:"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": template}],
        model="mixtral-8x7b-32768",
    )
    
    return chat_completion.choices[0].message.content

class Query(BaseModel):
    text: str

@app.post("/query")
async def process_query(query: Query) -> Dict[str, str]:
    try:
        question_embedding = get_query_embedding(query.text)
        query_result = query_index(question_embedding)

        docs = {x["metadata"]['text']: i for i, x in enumerate(query_result["matches"])}
        rerank_docs = rerank_documents(query.text, list(docs.keys()))
        reranked_texts = [doc.document.text for doc in rerank_docs.results]

        context = " ".join(reranked_texts)
        response = generate_response(context, query.text)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



