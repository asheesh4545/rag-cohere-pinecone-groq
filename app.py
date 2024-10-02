import os
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
import pinecone
import cohere
from groq import Groq


load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Initialize clients
# llm_groq = ChatGroq(temperature=0, model_name="groq/mixtral-8x7b-32768", api_key=GROQ_API_KEY)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
co = cohere.Client(COHERE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

def read_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    return " ".join(page.extract_text() for page in pdf.pages)

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=200)
    return text_splitter.split_text(text)

def embed_texts(texts):
    embeddings = FastEmbedEmbeddings()
    return embeddings.embed_documents(texts)

def create_or_get_index(index_name, dimension):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

def upsert_to_index(index, embeddings, texts):
    for i, (embedding, text) in enumerate(zip(embeddings, texts)):
        index.upsert([(str(i), embedding, {"text": text})])

def get_query_embedding(text):
    embeddings = FastEmbedEmbeddings()
    return embeddings.embed_query(text)

def query_index(index, query_embedding, top_k=5):
    return index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

def rerank_documents(query, documents, top_n=5):
    return co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=True
    )

def generate_response(context, query):
    template = f"Based on the following context: {context} generate precise summary related to question: {query} Do not remove necessary information related to context. Consider `\\n` as newline character."
    

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": template}],
        model="mixtral-8x7b-32768",
    )
    
    return chat_completion.choices[0].message.content

def main():
    # Read and process PDF
    pdf_text = read_pdf("iesc111.pdf")
    texts = split_text(pdf_text)
    embeddings = embed_texts(texts)

    # Create a Pinecone index in case it does not exists.
    index = create_or_get_index("testv2", len(embeddings[0]))


    if index.describe_index_stats()['total_vector_count'] == 0:
        upsert_to_index(index, embeddings, texts)
        print("Finished upserting to index.")
    else:
        print("Index already contains vectors. Skipping upsert.")

    # Query processing
    query = "Who is Heinrich Rudolph Hertz"
    question_embedding = get_query_embedding(query)
    query_result = query_index(index, question_embedding)

    # Rerank documents
    docs = {x["metadata"]['text']: i for i, x in enumerate(query_result["matches"])}
    rerank_docs = rerank_documents(query, list(docs.keys()))
    reranked_texts = [doc.document.text for doc in rerank_docs.results]

    # Generate response
    context = " ".join(reranked_texts)
    response = generate_response(context, query)
    
    print("------------------------------------------")
    print(response)

if __name__ == "__main__":
    main()