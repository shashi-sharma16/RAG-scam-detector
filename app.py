import chromadb
from sentence_transformers import SentenceTransformer

with open("data/scams.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunk_size = 200
chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

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

question = "A company wants me to pay a registeration fee before giving me a jon. Is this suspicious?"

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