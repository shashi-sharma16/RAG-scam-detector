import chromadb
import ollama
import re
from sentence_transformers import SentenceTransformer

with open("data/scams.txt", "r", encoding="utf-8") as file:
    text = file.read()

sentences = re.split(r'(?<=[.!?])\s+', text.strip())

chunks = []
current_chunk = ""

for sentence in sentences:
    if len(current_chunk) + len(sentence) <= 300:
        current_chunk += sentence + " "
    else:
        chunks.append(current_chunk.strip())
        current_chunk = sentence + " "

if current_chunk:
    chunks.append(current_chunk.strip())

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="Scam_knowledge"
)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("Stored", len(chunks), "chunks in ChromaDB.")

question = input("\nEnter your question: ")

question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

print("\nQuestion:")
print(question)

print("\nRetrieved information:")

for document in results["documents"][0]:
    print("\n---")
    print(document)

context = "\n\n".join(results["documents"][0])

prompt = f"""
You are a scam-awareness assistant.

Answer the user's question using only the information provided below.

Information:
{context}

User question:
{question}

Give a clear and simple answer.
Do not make up information that is not supported by the provided information.
"""

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAI Answer:")
print(response["message"]["content"])