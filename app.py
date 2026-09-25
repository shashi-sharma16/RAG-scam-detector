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

best_distance = results["distances"][0][0]

if best_distance < 1.2:
    match_level = "Strong"
elif best_distance < 1.5:
    match_level = "Moderate"
else:
    match_level = "Weak"

print("\nRetrieval Match:", match_level)

print("\nQuestion:")
print(question)

print("\nRetrieved information:")

for document in results["documents"][0]:
    print("\n---")
    print(document)

context = "\n\n".join(results["documents"][0])

prompt = f"""
You are a scam-awareness assistant.

IMPORTANT RULES:
1. Use ONLY the information provided in the Information section.
2. Do NOT add facts, assumptions, explanations, or advice that are not supported by the Information section.
3. If the Information section does not provide enough evidence, use "Unknown".
4. Keep the answer short and clear.
5. Follow the exact output format below.

Information:
{context}

User question:
{question}

Output format:

Scam Type: [Phishing / Fake Job Scam / Investment Scam / Other / Unknown]
Risk Level: [LOW / MEDIUM / HIGH / Unknown]

Reason:
[Explain briefly using only the provided information.]

Advice:
[Give practical advice based only on the provided information.]
"""

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    think=False
)

print("\nAI Answer:")
print(response["message"]["content"])