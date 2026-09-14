import chromadb
import ollama
import re
from sentence_transformers import SentenceTransformer

with open("data/scams.txt", "r", encoding="utf-8") as file:
    text = file.read()

text = re.sub(r'\s+', ' ', text).strip()
sentences = re.split(r'(?<=[.!?])\s+', text)

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

if collection.count() == 0:
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )
    print("Stored", len(chunks), "chunks in ChromaDB.")
else:
    print("Chunks already exist in ChromaDB.")

question = input("\nEnter your question: ")

question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2,
    include=["documents", "distances"]
)

print("\nDistances:", results["distances"][0])

print("\nQuestion:")
print(question)

print("\nRetrieved information:")

for document in results["documents"][0]:
    print("\n---")
    print(document)

context = "\n\n".join(results["documents"][0])

prompt = f"""
You are a scam-awareness assistant.

Analyze the user's question using only the information provided below.

Information:
{context}

User question:
{question}

Give your response in this format:

Scam Type: [Phishing / Fake Job Scam / Investment Scam / Other / Unknown]
Risk Level: [LOW / MEDIUM / HIGH]

Reason:
[Explain briefly why the situation may or may not be suspicious.]

Advice:
[Give a short and practical safety recommendation.]

Do not make up information that is not supported by the provided information.
If the information is not enough to identify the scam type, write "Unknown".
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