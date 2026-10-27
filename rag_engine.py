
import re

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# 1. Load the scam knowledge base
with open("data/scams.txt", "r", encoding="utf-8-sig") as file:
    knowledge_text = file.read()

clean_text = re.sub(r"\s+", " ", knowledge_text).strip()
sentences = re.split(r"(?<=[.!?])\s+", clean_text)

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


# 2. Load the embedding model and ChromaDB
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
        ],
    )


# 3. Recognize known warning signs
WARNING_SIGNS = {
    "Phishing Scam": [
        "password", "banking credentials", "bank details", "otp",
        "one time password", "credit card", "debit card",
        "suspicious link", "unfamiliar link", "unknown link",
        "click this link", "click a link", "account blocked",
        "account suspended", "account will be blocked",
        "verify your account", "verify your identity", "urgent",
        "act immediately", "share your otp", "enter your password",
    ],
    "Fake Job Scam": [
        "high salary", "high salaries", "huge salary",
        "high-paying job", "high paying job", "easy money",
        "registration fee", "registration fees", "training fee",
        "training fees", "security deposit", "joining fee",
        "application fee", "pay money", "pay a fee",
        "pay before joining", "pay to get the job",
        "money to get the job", "job offer fee",
    ],
    "Investment Scam": [
        "guaranteed returns", "guaranteed high returns",
        "guaranteed profit", "guaranteed profits", "high returns",
        "unusually high returns", "huge profits", "little or no risk",
        "no risk", "risk-free investment", "invest quickly",
        "invest immediately", "limited-time investment",
        "double your money", "double my money",
        "double your investment", "transfer money",
        "personal account", "assured returns", "assured profit",
    ],
    "Impersonation Scam": [
        "pretending to be a bank employee",
        "pretending to be a government official",
        "pretending to be police", "bank employee",
        "government official", "police officer",
        "company representative", "claiming to be from the bank",
        "claiming to represent the bank", "verify their identity",
        "unverified identity", "urgent problem with my account",
        "asked for my otp", "asking for my otp",
    ],
    "Payment Fraud": [
        "payment screenshot", "fake payment screenshot",
        "money has not arrived", "money hasn't arrived",
        "money did not arrive", "money didn't arrive",
        "release goods", "qr code to receive money",
        "scan a qr code", "payment request",
        "payment confirmation", "proof of payment",
        "payment is complete", "payment not received",
    ],
    "Romance Scam": [
        "online relationship", "met someone online", "met online",
        "romantic relationship", "says they love me",
        "asking me for money", "asks me for money",
        "asking for money", "send money", "medical emergency",
        "travel expenses", "send money to my partner",
        "keep it secret", "relationship and money",
        "never met in person", "online romantic partner",
    ],
}


# 4. Extract advice from the matching knowledge-base sections
def get_advice(categories):
    advice_lines = []

    for category in categories:
        section = re.search(
            rf"Scam Type:\s*{re.escape(category)}\b(.*?)(?=Scam Type:|$)",
            clean_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not section:
            continue

        advice_match = re.search(
            r"\bAdvice:\s*(.*)",
            section.group(1),
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not advice_match:
            continue

        items = re.split(r"\s+-\s+", advice_match.group(1))

        for item in items:
            item = item.strip().lstrip("- ").strip()

            if item and item not in advice_lines:
                advice_lines.append(item)

    return advice_lines


# 5. Analyze one message and return structured results
def analyze_message(question):
    question = question.strip()

    if not question:
        raise ValueError("Please enter a message to analyze.")

    question_lower = question.lower()

    matched_signs = {
        category: [
            sign for sign in signs if sign in question_lower
        ]
        for category, signs in WARNING_SIGNS.items()
    }

    matched_signs = {
        category: signs
        for category, signs in matched_signs.items()
        if signs
    }

    categories = list(matched_signs.keys())

    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(8, collection.count()),
        include=["documents", "distances", "metadatas"],
    )

    best_distance = results["distances"][0][0]
    sources = results["metadatas"][0]

    if best_distance < 1.2:
        match_level = "Strong"
    elif best_distance < 1.5:
        match_level = "Moderate"
    else:
        match_level = "Weak"

    # Do not ask the model to analyze unrelated material.
    if best_distance >= 1.5 or not categories:
        return {
            "scam_type": "Unknown",
            "risk_level": "Unknown",
            "reason": (
                "There is not enough recognized evidence and relevant "
                "knowledge-base information to classify this message."
            ),
            "warning_signs": [],
            "advice": [],
            "sources": sources,
            "retrieval_match": match_level,
        }

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
You are a scam-awareness assistant.

Use ONLY the supplied information and user message.
Do not invent facts, warning signs, or advice.
Consider these detected categories: {", ".join(categories)}.
Report all these categories, not just the first one.
Explain briefly which message details support the assessment.
Do not write the Advice section; Python supplies advice separately.

Information:
{context}

User message:
{question}

Output exactly in this format:

Reason: A short explanation based only on the information.
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        think=False,
    )

    answer = response["message"]["content"].strip()

    reason_match = re.search(
        r"Reason:\s*(.*)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )

    reason = reason_match.group(1).strip() if reason_match else answer

    total_signs = sum(len(signs) for signs in matched_signs.values())

    if total_signs >= 2:
        risk_level = "HIGH"
    else:
        risk_level = "MEDIUM"

    all_warning_signs = [
        f"{category}: {sign}"
        for category, signs in matched_signs.items()
        for sign in signs
    ]

    return {
        "scam_type": ", ".join(categories),
        "risk_level": risk_level,
        "reason": reason,
        "warning_signs": all_warning_signs,
        "advice": get_advice(categories),
        "sources": sources,
        "retrieval_match": match_level,
    }
