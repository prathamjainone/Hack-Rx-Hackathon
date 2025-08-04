import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai

app = FastAPI()

# Configure Gemini API key (using your provided key)
genai.configure(api_key='AIzaSyCR2RHfatumqAQwHerRQuEEUWKf1QAc76M')
gemini_model = "gemini-1.5-pro"  # Or "gemini-2.5-pro" if available on your account

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def semantic_search(query, chunk_embeddings, chunks, top_k=3):
    query_embedding = embedder.encode([query])[0]
    similarities = np.dot(chunk_embeddings, query_embedding)
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def ask_gemini(question, contexts):
    combined_context = "\n\n".join(contexts)
    prompt = (
        f"Answer the question using ONLY the information from the policy clauses below.\n"
        f"If the answer is not in the text, say 'Not specified in the document.'\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {question}\n"
        f"Answer concisely:"
    )
    response = genai.GenerativeModel(gemini_model).generate_content(prompt)
    return response.text.strip()

@app.post("/hackrx/run")
async def hackrx_run(request: QueryRequest, authorization: Optional[str] = Header(None)):
    if authorization != "Bearer 90c5341ca9a6ee22ff14c359daedbed013562397283029e7cf061e5b8d40":
        raise HTTPException(status_code=401, detail="Unauthorized")

    pdf_url = request.documents
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download document")

    pdf_bytes = BytesIO(response.content)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    chunks = chunk_text(full_text)
    chunk_embeddings = embedder.encode(chunks)

    answers = []
    for question in request.questions:
        best_chunks = semantic_search(question, chunk_embeddings, chunks, top_k=3)
        answer = ask_gemini(question, best_chunks)
        answers.append(answer)

    return {"answers": answers, "success": True, "message": "PDF processed"}

