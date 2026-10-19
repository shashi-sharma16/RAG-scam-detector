
import chromadb
import ollama
import re
from sentence_transformers import SentenceTransformer


# 1. Read and clean the knowledge base
with open("data/scams.txt", "r", encoding="utf-8-sig") as file:
    text = file.read()

text = re.sub(r"\s+", " ", text).strip()
sentences = re.split(r"(?<=[.!?])\s+", text)


# 2. Split text into chunks
chunks = []
current_chunk = ""

for sentence in sentences:
    if len(current_chunk) + len(sentence) <= 300:
        current_chunk += sentence + " "
    else:
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        current_chunk = sentence + " "

if current_chunk.strip():
    chunks.append(current_chunk.strip())


# 3. Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)


# 4. Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="Scam_knowledge"
)


# 5. Store chunks if the collection is empty
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


# 6. Get the user's question
question = input("\nEnter your question: ")

question_embedding = model.encode(question).tolist()


# 7. Retrieve relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2,
    include=["documents", "distances", "metadatas"]
)

best_distance = results["distances"][0][0]


# 8. Display retrieval match
if best_distance < 1.2:
    match_level = "Strong"
elif best_distance < 1.5:
    match_level = "Moderate"
else:
    match_level = "Weak"

print("\nRetrieval Match:", match_level)

context = "\n\n".join(results["documents"][0])
sources = results["metadatas"][0]


# 9. Handle weak retrieval
if best_distance >= 1.5:
    print("\nNo relevant information found in the knowledge base.")
    print("\n" + "=" * 40)
    print("           SCAM ANALYSIS")
    print("=" * 40)
    print("Scam Type: Unknown")
    print("Risk Level: Unknown")
    print("\nReason:")
    print(
        "The knowledge base does not contain enough information "
        "to analyze this question."
    )
    print("\nAdvice:")
    print("Unknown")
    print("=" * 40)
    raise SystemExit


# 10. Check whether the question contains a known warning sign
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
    ],
}

matched_category = None

for category, signs in warning_signs.items():
    if any(sign in question_lower for sign in signs):
        matched_category = category
        break


# 11. Handle questions without a recognized warning sign
if matched_category is None:
    print("\n" + "=" * 40)
    print("           SCAM ANALYSIS")
    print("=" * 40)
    print("Scam Type: Unknown")
    print("Risk Level: Unknown")
    print("\nReason:")
    print(
        "The retrieved information does not provide enough "
        "evidence to analyze this situation."
    )
    print("\nAdvice:")
    print("Unknown")
    print("=" * 40)
    raise SystemExit


# 12. Generate the scam analysis
prompt = f"""
You are a scam-awareness assistant.

Analyze the user's question using ONLY the provided Information.

IMPORTANT RULES:
1. Use only facts explicitly stated in the Information.
2. Do not add outside knowledge.
3. Do not invent facts, warning signs, or advice.
4. Identify a scam type only when the Information supports it.
5. Use Unknown if there is insufficient evidence for a field.
6. Assign HIGH risk when multiple strong warning signs are present.
7. Assign MEDIUM risk when a warning sign is present but evidence is limited.
8. Keep the answer short and clear.
9. Do not include the Advice section in your response.
10. Follow the exact output format.

Information:
{context}

User question:
{question}

Output format:

Scam Type: [Phishing / Fake Job Scam / Investment Scam / Other / Unknown]
Risk Level: [LOW / MEDIUM / HIGH / Unknown]

Reason:
[Explain briefly using only the Information.]
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


# 13. Extract advice directly from retrieved knowledge
advice_lines = []

for chunk in results["documents"][0]:
    match = re.search(
        r"\bAdvice:\s*(.*)",
        chunk,
        flags=re.IGNORECASE
    )

    if match:
        advice_part = match.group(1)

        # Split bullet points whether they are on one line or several lines
        items = re.split(r"\s+-\s+", advice_part)

        for item in items:
            item = item.strip()

            if item:
                advice_lines.append(item)

# Remove duplicate advice while preserving order
advice_lines = list(dict.fromkeys(advice_lines))

if advice_lines:
    grounded_advice = "\n".join(
        f"- {advice.lstrip('- ').strip()}"
        for advice in advice_lines
)
else:
    grounded_advice = "Unknown"


# 14. Display the final analysis
answer = response["message"]["content"].strip()

# Remove any Advice section generated by the model
answer = re.split(
    r"\n\s*Advice\s*:",
    answer,
    maxsplit=1,
    flags=re.IGNORECASE
)[0].strip()

print("\n" + "=" * 40)
print("           SCAM ANALYSIS")
print("=" * 40)

print(answer)

print("\nAdvice:")
print(grounded_advice)

print("\nSources:")

for i, source in enumerate(sources, start=1):
    print(f"[{i}] {source['source']} - Chunk {source['chunk']}")

print("=" * 40)
