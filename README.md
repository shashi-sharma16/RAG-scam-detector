# RAG Scam Detector

**A local AI-powered scam message analysis system using Retrieval-Augmented Generation (RAG).**

RAG Scam Detector is a Python-based project that analyzes suspicious messages, identifies recognized scam warning signs, retrieves relevant information from a local scam-awareness knowledge base, and provides explanations, risk assessments, and preventive advice.

## Features

- Detects six scam categories: Phishing, Fake Job, Investment, Impersonation, Payment Fraud, and Romance scams.
- Assigns risk levels: LOW, MEDIUM, HIGH, or CRITICAL.
- Identifies recognized warning signs in messages.
- Uses Retrieval-Augmented Generation (RAG) to retrieve relevant scam-awareness information.
- Generates explanations using a locally running language model.
- Provides category-specific preventive advice.
- Includes a web dashboard built with Flask.
- Uses Sentence Transformers for embeddings and ChromaDB for persistent vector storage.
- Includes an evaluation framework with a dataset of 60 synthetic messages.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Flask | Web application |
| Ollama (`qwen2.5:3b`) | Local language model |
| ChromaDB | Vector database |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Text embeddings |
| HTML, CSS, JavaScript | Dashboard interface |

## How It Works

1. The user submits a message through the web dashboard.
2. Python checks the message for recognized scam warning signs.
3. The message is converted into an embedding.
4. ChromaDB retrieves relevant information from the scam-awareness knowledge base.
5. Ollama generates an explanation using the retrieved context.
6. The application presents the detected categories, risk level, warning signs, preventive advice, and source information.

## Evaluation

The project includes a synthetic evaluation dataset of 60 messages covering scam categories and legitimate messages.

### Latest Results

| Metric | Result |
|---|---:|
| Total test messages | 60 |
| Binary accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-score | 100% |
| Exact category matches | 39/45 |
| Strict overall matches | 54/60 |

**Important:** These results apply only to the current synthetic dataset. The dataset was used during development and rule adjustments, so these figures should not be interpreted as independent or real-world performance measurements.

The evaluation script calculates binary scam-detection metrics and category-classification results. Evaluation outputs are saved to `evaluation/evaluation_results.csv`.

To run the evaluation:

```powershell
python evaluation/run_evaluation.py
```

## Project Structure

```text
RAG Scam Detector/
├── data/
│   └── scams.txt
├── templates/
│   └── index.html
├── evaluation/
│   ├── evaluation_dataset.csv
│   ├── evaluation_results.csv
│   ├── run_evaluation.py
│   └── run_evaluation_backup.py
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
- [Ollama](https://ollama.com/)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/shashi-sharma16/RAG-scam-detector.git
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

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the language model

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running before analyzing messages.

### 5. Start the web application

```bash
python web_app.py
```

Open the local address shown in the terminal, usually:

`http://127.0.0.1:5000`

The embedding model may download automatically the first time it is loaded.

## Limitations

- Detection depends on the warning signs defined in the code and the information available in the knowledge base.
- Risk levels are heuristic assessments, not verified probabilities of fraud.
- Retrieval quality depends on the knowledge base and similarity thresholds.
- A message classified as `Unknown` or assigned a low risk level is not necessarily safe.
- The evaluation dataset is synthetic and does not establish real-world detection accuracy.
- This is an educational project, not a substitute for independent verification or professional advice.

## Future Improvements

- Expand the scam-awareness knowledge base.
- Improve contextual scam classification and reduce incorrect additional category predictions.
- Evaluate performance on independent, previously unseen messages.
- Improve risk-level consistency.
- Add automated tests and improved error handling.
- Explore URL analysis and additional scam-detection techniques.

## Author

**Shashi Sharma**

Second-year Data Science student interested in backend development, web development, and cybersecurity.