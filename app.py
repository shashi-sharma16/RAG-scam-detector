
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
    n_results=min(8, collection.count()),
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
        "bank details",
        "otp",
        "one time password",
        "credit card",
        "debit card",
        "suspicious link",
        "unfamiliar link",
        "unknown link",
        "click this link",
        "click a link",
        "account blocked",
        "account suspended",
        "account will be blocked",
        "verify your account",
        "verify your identity",
        "urgent",
        "act immediately",
        "share your otp",
        "enter your password",
    ],
    "Fake Job Scam": [
        "high salary",
        "high salaries",
        "huge salary",
        "high-paying job",
        "high paying job",
        "easy money",
        "registration fee",
        "registration fees",
        "training fee",
        "training fees",
        "security deposit",
        "joining fee",
        "application fee",
        "pay money",
        "pay a fee",
        "pay before joining",
        "payment",
        "pay to get the job",
        "money to get the job",
        "job offer fee",
    ],
    "Investment Scam": [
        "guaranteed returns",
        "guaranteed high returns",
        "guaranteed profit",
        "guaranteed profits",
        "high returns",
        "unusually high returns",
        "huge profits",
        "little or no risk",
        "no risk",
        "risk-free investment",
        "invest quickly",
        "invest immediately",
        "limited-time investment",
        "double your money",
        "transfer money",
        "personal account",
        "assured returns",
        "assured profit",
        "double my money",
        "double your investment",
    ],
}

matched_categories = []
matched_signs_by_category = {}

for category, signs in warning_signs.items():
    matched_signs = [
        sign for sign in signs
        if sign in question_lower
    ]

    if matched_signs:
        matched_categories.append(category)
        matched_signs_by_category[category] = matched_signs

matched_category = (
    matched_categories[0]
    if len(matched_categories) == 1
    else None
)

# 11. Handle questions without a recognized warning sign
if not matched_categories:
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
    print("\nWarning Signs Detected:")
    print("No recognized warning signs detected.")
    print("\nAdvice:")
    print("Unknown")
    print("=" * 40)
    raise SystemExit

# Prepare the categories detected from the user's question
if len(matched_categories) == 1:
    category_instruction = matched_categories[0]
else:
    category_instruction = ", ".join(matched_categories)

# 12. Generate the scam analysis

prompt = f"""
You are a scam-awareness assistant.

Analyze the user's question using ONLY the provided Information
and the detected warning-sign categories.

IMPORTANT RULES:
1. Use only facts explicitly stated in the Information.
2. Do not add outside knowledge.
3. Do not invent warning signs or advice.
4. Consider these categories detected from the user's question:
   {category_instruction}
5. If one category is detected, report that category if supported
   by the Information.
6. If multiple categories are detected, list all supported categories.
7. Do not force a single category when multiple are supported.
8. Assign HIGH risk when multiple strong warning signs are present.
9. Assign MEDIUM risk when warning signs exist but evidence is limited.
10. Use Unknown if there is insufficient evidence.
11. Keep the answer short and clear.
12. Do not include the Advice section in your response.
13. Do not put square brackets around the output values.
14. Follow the exact output format.

Information:
{context}

User question:
{question}

Output format:

Scam Type: [supported scam category or categories, or Unknown]
Risk Level: [LOW / MEDIUM / HIGH / Unknown]

Reason:
[Brief explanation based only on the Information.]
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


# 13. Extract advice directly from the complete knowledge base
advice_lines = []

for category in matched_categories:
    # Find the complete section for this scam category
    section_match = re.search(
        rf"Scam Type:\s*{re.escape(category)}\b(.*?)(?=Scam Type:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not section_match:
        continue

    # Extract the Advice section from that category
    advice_match = re.search(
        r"\bAdvice:\s*(.*)",
        section_match.group(1),
        flags=re.IGNORECASE | re.DOTALL
    )

    if not advice_match:
        continue

    advice_part = advice_match.group(1)

    # Separate individual advice bullets
    items = re.split(r"\s+-\s+", advice_part)

    for item in items:
        item = item.strip().lstrip("- ").strip()

        if item and item not in advice_lines:
            advice_lines.append(item)

if advice_lines:
    grounded_advice = "\n".join(
        f"- {advice}" for advice in advice_lines
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

# Ensure all detected scam categories appear in the final answer
if matched_categories:
    detected_type = ", ".join(matched_categories)

    answer = re.sub(
        r"(?im)^Scam Type\s*:.*$",
        f"Scam Type: {detected_type}",
        answer,
        count=1
    )

print("\n" + "=" * 40)
print("           SCAM ANALYSIS")
print("=" * 40)

print(answer)

# Display warning signs detected in the user's question
print("\nWarning Signs Detected:")

if matched_signs_by_category:
    for category, signs in matched_signs_by_category.items():
        print(f"{category}:")
        for sign in signs:
            print(f"- {sign.capitalize()}")
else:
    print("No recognized warning signs detected.")

print("\nAdvice:")
print(grounded_advice)

print("\nSources:")

for i, source in enumerate(sources, start=1):
    print(f"[{i}] {source['source']} - Chunk {source['chunk']}")

print("=" * 40)
