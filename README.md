# PathRAG Document Analysis System

This is a powerful, end-to-end Question & Answering application designed to analyze complex technical documents. It uses a sophisticated Knowledge Graph-based approach (PathRAG) to provide accurate, context-aware answers.

This implementation has been specifically engineered to run on free, open-source models, providing a powerful yet cost-effective solution for document intelligence.

## Key Features

- **Knowledge Graph Construction:** Automatically reads a PDF document and builds a detailed knowledge graph of its technical entities and their relationships.
- **Hybrid Q&A:** Intelligently answers both highly specific questions (e.g., "What is requirement SWS_RTE_01001?") and broad, general questions (e.g., "What is the scope of this document?").
- **Hallucination-Resistant:** The system is designed to state when it cannot find an answer in the document, rather than making up incorrect information.
- **Interactive UI:** A simple web interface built with Gradio for asking questions and managing the indexing process.
- **Data Management:** Allows users to index new documents (wiping old data) and create downloadable backups of the knowledge base.

## Setup and Usage Guide

This guide provides a step-by-step process to set up and run the application.

### Prerequisites
- Python 3.10 or newer
- Git

### Step 1: Clone the Repository
First, get a copy of the project on your local machine.
```bash
git clone https://github.com/YourUsername/YourNewRepoName.git
cd YourNewRepoName
```
### Step 2: Set Up a Virtual Environment (Highly Recommended)
This creates an isolated environment for the project's dependencies.

### Create the environment
```
python -m venv venv
```
### Activate it (on Windows)
```
venv\Scripts\activate
```
### Activate it (on Mac/Linux)
```
source venv/bin/activate
```
### Step 3: Install Required Libraries
Install all the necessary Python packages using the provided file.
```python
pip install -r requirements.txt
```
### Step 4: Add Your API Key
This application uses the Ollama Turbo API for high-quality AI reasoning.
Create a file named .env in the root of the project.
Add your Ollama API key to this file like so:
```OLLAMA_API_KEY="ollama_your_api_key_here"```
(Note: For deployment on platforms like Hugging Face Spaces, this key should be stored as a secure "Secret".)

### Step 5: Run the Application
You are now ready to start the web interface.
```python main.py```

After a few moments, you will see a local URL in your terminal (like http://127.0.0.1:7860). Open this link in your web browser to use the application.

### How the Application Works
The application has two main workflows accessible through the UI tabs.

### Workflow A: Indexing a New Document
This is the process of teaching the AI about a new document. This is a one-time, slow process for each new document.
- **Navigate to the "Admin & Indexing" tab.**
- **Under "Index a New Document," click to upload your PDF file.**
- **Click the "Wipe and Start Indexing" button. This will delete any old knowledge and begin the indexing process.**
- **You can monitor the progress in the terminal. This may take a significant amount of time (30-60 minutes) depending on the document size.**

### Workflow B: Asking Questions
Once a document has been indexed, you can ask questions.
- **Navigate to the "Q&A" tab.**
- **Type your question into the "Ask a Question" box and press Enter.**
- **The system will use its knowledge graph to find the most relevant information and generate an answer. The first query may be slow as the models are loaded into memory.**

### Project Structure Overview
```
├── documents/              # Place your source PDFs here (optional, for local use)
├── autosar_rag_data/       # Stores the permanent, pre-built knowledge base
├── PathRAG/                # The core PathRAG library
│   ├── __init__.py
│   ├── base.py             # Abstract base classes for storage
│   ├── llm.py              # Functions for interacting with AI models
│   ├── operate.py          # Core logic for indexing and querying
│   ├── pathrag.py          # The main PathRAG class and configuration
│   ├── prompt.py           # All prompts used by the LLM
│   └── storage.py          # Concrete storage implementations
├── main.py                 # The main Gradio application file
├── requirements.txt        # All Python dependencies
└── README.md               # This file
```
## Cititation

```python
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}

```





