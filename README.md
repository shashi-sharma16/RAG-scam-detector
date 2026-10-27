# RAG Scam Detector

**RAG-powered scam message analysis using a local language model.**

RAG Scam Detector is a Python-based project that analyzes suspicious messages, identifies recognized scam warning signs, retrieves relevant information from a local scam-awareness knowledge base, and provides risk assessments and preventive advice.

## Features

- Detects six scam categories: Phishing, Fake Job, Investment, Impersonation, Payment Fraud, and Romance scams.
- Assigns risk levels: LOW, MEDIUM, HIGH, or CRITICAL.
- Identifies recognized warning signs in user messages.
- Uses Retrieval-Augmented Generation (RAG) to retrieve relevant scam-awareness information.
- Generates explanations using a locally running language model.
- Provides category-specific preventive advice from the knowledge base.
- Includes a web dashboard built with Flask.
- Uses local embeddings and a persistent ChromaDB vector database.

## Tech Stack

- **Language:** Python
- **Web Framework:** Flask
- **Vector Database:** ChromaDB
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Local LLM:** Ollama (`qwen2.5:3b`)
- **Frontend:** HTML, CSS, JavaScript

## How It Works

1. The user submits a message through the web dashboard.
2. Python checks the message for recognized scam warning signs.
3. The message is converted into an embedding and compared with the stored knowledge base.
4. Relevant information is retrieved from ChromaDB.
5. Ollama generates a short explanation using the retrieved context.
6. The application returns the detected scam category, risk level, warning signs, preventive advice, and source information.


## Project Structure

```text
RAG Scam Detector/
├── data/
│   └── scams.txt
├── templates/
│   └── index.html
├── chroma_db/          # Generated locally
├── app.py              # Earlier standalone version
├── rag_engine.py
├── web_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `chroma_db/` directory stores the persistent vector database and is generated locally when the application initializes it.

## Setup and Installation

### Prerequisites

- Python 3.13 or another compatible Python version
- Ollama installed and running
- Git

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd "RAG Scam Detector"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the local language model

```bash
ollama pull qwen2.5:3b
```

Ensure the Ollama application is running before analyzing messages.

### 5. Start the application

```bash
python web_app.py
```

Open the local address shown in the terminal, usually:

`http://127.0.0.1:5000`

The embedding model may download automatically the first time it is loaded.

## Limitations

- Scam classification depends on the warning signs defined in the code and the information available in the knowledge base.
- Risk levels are heuristic assessments, not verified probabilities of fraud.
- Retrieval quality depends on the knowledge base and similarity thresholds.
- A message classified as Unknown or assigned a low risk level is not necessarily safe.
- The application is an educational project and should not replace independent verification or professional advice.

## Future Improvements

- Expand the scam-awareness knowledge base.
- Improve contextual detection of scam indicators.
- Add evaluation datasets and precision, recall, and F1-score measurements.
- Add automated testing and improved error handling.
- Explore URL analysis and additional scam-detection techniques.

## Author

**Shashi Sharma**

Second-year Data Science student interested in backend development, web development, and cybersecurity.
