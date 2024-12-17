import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

# Initialize the LLM (GPT model via OpenAI API)
openai.api_key = 'your-openai-api-key'

# Initialize the SentenceTransformer model for text embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimensionality of the embeddings (for 'all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(dimension)
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

# Function to chunk the extracted text into smaller parts
def chunk_text(text, chunk_size=500):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to embed text using the SentenceTransformer model
def embed_text(text_chunks):
    embeddings = embedding_model.encode(text_chunks)
    return embeddings

# Function to add embeddings to FAISS index
def add_to_faiss_index(embeddings):
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)

# Function to perform a similarity search in the FAISS index
def search_in_faiss(query, top_k=5):
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]

# Function to generate response using OpenAI's GPT model
def generate_response(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the following question using the context provided:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = openai.Completion.create(
        engine="gpt-4", 
        prompt=prompt, 
        max_tokens=200, 
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Main function to process PDF files, store embeddings, and respond to queries
def process_pdfs_and_query(pdf_files, query):
    all_chunks = []
    
    # Step 1: Data Ingestion
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    
    # Step 2: Embed chunks and add to FAISS index
    embeddings = embed_text(all_chunks)
    add_to_faiss_index(embeddings)
    
    # Step 3: Handle User Query
    indices, distances = search_in_faiss(query)
    relevant_chunks = [all_chunks[i] for i in indices]
    
    # Step 4: Generate Response using LLM (GPT)
    response = generate_response(query, relevant_chunks)
    
    return response

# Example Usage:
pdf_files = ["document1.pdf", "document2.pdf"]  
query = "What are the key differences between product A and product B?"

response = process_pdfs_and_query(pdf_files, query)
print(response)
