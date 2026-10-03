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
    embeddings=embeddings.tolist(),
    metadatas=[
        {"source": "scams.txt", "chunk": i + 1}
        for i in range(len(chunks))
    ]
)
    print("Stored", len(chunks), "chunks in ChromaDB.")
else:
    print("Chunks already exist in ChromaDB.")

question = input("\nEnter your question: ")

question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2,
    include=["documents", "distances", "metadatas"]
)

best_distance = results["distances"][0][0]

if best_distance < 1.2:
    match_level = "Strong"
elif best_distance < 1.5:
    match_level = "Moderate"
else:
    match_level = "Weak"

print("\nRetrieval Match:", match_level)

context = "\n\n".join(results["documents"][0])
sources = results["metadatas"][0]

if best_distance >= 1.5:
    print("\nNo relevant information found in the knowledge base.")

    print("\n" + "=" * 40)
    print("           SCAM ANALYSIS")
    print("=" * 40)

    print("Scam Type: Unknown")
    print("Risk Level: Unknown")

    print("\nReason:")
    print("The knowledge base does not contain enough information to analyze this question.")

    print("\nAdvice:")
    print("Unknown")

    print("=" * 40)

    exit()

prompt = f"""
You are a scam-awareness assistant.

Your job is to analyze the user's question using ONLY the provided Information.

IMPORTANT RULES:
1. Use ONLY facts explicitly stated in the Information.
2. Do NOT assume that a detail mentioned in the question is true unless the Information supports it.
3. Do NOT add outside knowledge from your own training.
4. Do NOT invent facts, warning signs, causes, or advice.
5. If the Information does not provide enough evidence to identify a scam type, write "Unknown".
6. If the Information does not provide enough evidence for a risk level, write "Unknown".
7. If the Information does not support specific advice, write "Unknown".
8. The Reason must mention only facts supported by the Information.
9. Keep the answer short and clear.
10. Follow the exact output format.

Information:
{context}

User question:
{question}

Output format:

Scam Type: [Phishing / Fake Job Scam / Investment Scam / Other / Unknown]
Risk Level: [LOW / MEDIUM / HIGH / Unknown]

Reason:
[Explain briefly using ONLY the Information.]

Advice:
[Give relevant advice from the Information. If no relevant advice is provided, write "Unknown".]
"""

question_lower = question.lower()

warning_signs = {
    "Phishing": [
        "password",
        "banking credentials",
        "otp",
        "credit card",
        "suspicious link",
        "unfamiliar link",
        "account blocked",
        "account suspended",
        "click a link",
        "urgent",
    ],

    "Fake Job Scam": [
        "high salary",
        "high salaries",
        "registration fee",
        "training fee",
        "security deposit",
        "pay money",
        "pay a fee",
        "payment",
    ],

    "Investment Scam": [
        "guaranteed returns",
        "guaranteed high returns",
        "high returns",
        "unusually high returns",
        "little or no risk",
        "invest quickly",
        "invest immediately",
        "transfer money",
        "personal account",
    ]
}

matched_category = None

for category, signs in warning_signs.items():
    for sign in signs:
        if sign in question_lower:
            matched_category = category
            break

    if matched_category:
        break

if matched_category is None:
    print("\n" + "=" * 40)
    print("           SCAM ANALYSIS")
    print("=" * 40)

    print("Scam Type: Unknown")
    print("Risk Level: Unknown")

    print("\nReason:")
    print("The retrieved information does not provide enough evidence to analyze this situation.")

    print("\nAdvice:")
    print("Unknown")

    print("=" * 40)

    exit()

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

# Extract advice directly from retrieved knowledge
advice_lines = []

for chunk in results["documents"][0]:
    if "Advice:" in chunk:
        advice_part = chunk.split("Advice:", 1)[1]

        for line in advice_part.splitlines():
            line = line.strip()

            if line.startswith("-"):
                advice_lines.append(line[1:].strip())

if advice_lines:
    grounded_advice = "\n".join(
        f"- {advice}" for advice in advice_lines
    )
else:
    grounded_advice = "Unknown"

print("\n" + "=" * 40)
print("           SCAM ANALYSIS")
print("=" * 40)

answer = response["message"]["content"]

# Remove the LLM-generated Advice section
if "Advice:" in answer:
    answer = answer.split("Advice:", 1)[0].strip()

print(answer)

print("\nAdvice:")
print(grounded_advice)

print("\nSources:")

for i, source in enumerate(sources, start=1):
    print(f"[{i}] {source['source']} - Chunk {source['chunk']}")

print("=" * 40)