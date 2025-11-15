# 📚 Retrieval-Augmented Generation (RAG) Pipeline – Internship Project

### **Software Developer Intern (AI Team), Ailaysa – Taramani**

This project was developed during my internship at **Ailaysa**, an AI-driven language translation & NLP startup.
My role in the **AI Team** was focused on building a fully functional **RAG (Retrieval-Augmented Generation) pipeline** that converts PDF files into a searchable vector database and enables accurate, context-grounded LLM responses.

This repository contains the core RAG pipeline components that I implemented.

---

## 🚀 **Project Overview**

The goal of this work was to create a reliable system that allows an AI model to answer user questions **strictly based on uploaded documents**, reducing hallucinations and improving factual accuracy.

The RAG pipeline performs the following:

1. **PDF Document Loading & Chunking**
2. **Sentence Transformer Embeddings Generation**
3. **FAISS Vector Store Creation (document indexing)**
4. **Context Retrieval for Questions (Top-K Matching)**
5. **LLM Response Generation using Ollama (Qwen2.5)**
6. **Cited, Context-Aware Answers**

---

## 🧩 **Architecture**

```
PDF → PyPDFLoader → Text Chunks
        ↓
Embeddings (HuggingFace - mpnet-base-v2)
        ↓
FAISS Vector Store (.pkl)
        ↓
User Query → Retriever → Context
        ↓
LLM Prompting (Ollama - Qwen2.5)
        ↓
Final Answer + Citations
```

---

## 🔧 **Technologies Used**

### **Core RAG Components**

* **LangChain**
* **FAISS Vector Store**
* **HuggingFace Embeddings (all-mpnet-base-v2)**
* **PyPDFLoader** (PDF parsing)
* **Qwen2.5 via Ollama** (LLM inference)

### **Languages**

* Python 3.x

---

## 📂 Folder Structure

```
📦 rag-pipeline/
 ┣ 📜 rag_pipeline.py
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┗ 📂 example_inputs/
       ┗ sample.pdf
```

---

## 🧠 **Pipeline Steps (What I Built)**

### ✔ 1. **PDF Loading & Text Extraction**

```python
loader = PyPDFLoader(pdf_path)
documents = loader.load()
```

### ✔ 2. **Convert Text → Vector Embeddings**

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)
```

### ✔ 3. **Create FAISS Vector Store**

```python
db = FAISS.from_documents(documents, embeddings)
```

### ✔ 4. **Save Index as Pickle File**

```python
with open("leader_data.pkl", "wb") as f:
    pickle.dump(db, f)
```

### ✔ 5. **RAG Query → Retrieval + LLM Generation**

```python
retriever = db.as_retriever(search_kwargs={"k": 3})
context_docs = retriever.get_relevant_documents(user_query)
```

### ✔ 6. **Ollama (Qwen2.5) Response Generation**

```python
llm = Ollama(model="qwen2.5", base_url="http://localhost:11434")
response = llm.invoke(prompt_with_context)
```

---

## 🧪 Example Prompt Used in the RAG Pipeline

```txt
You are a helpful assistant. Answer based ONLY on the provided context.

RULES:
- Do NOT hallucinate.
- If the answer can't be found, say "The context does not contain this information."
- Keep the answer clear and factual.

Context:
{context}

User Question:
{input}

Answer:
```

---

## 🌟 **My Contribution (Internship Work)**

During my internship, I implemented:

### 🔹 **Complete RAG Pipeline**

* PDF loading
* Document chunking
* Text embedding (MPNet model)
* FAISS vector store creation
* Query-based context retrieval

### 🔹 **LLM Integration**

* Connected **Ollama (Qwen2.5)** for inference           https://www.prismetric.com/qwen-2-5-what-it-is-and-how-to-use-it/
* Designed prompts to prevent hallucination and ensure accuracy

### 🔹 **Document-grounded Conversation Logic**

* Strict context enforcement
* Citations extracted from FAISS metadata
* Fallback messages (e.g., unclear query, short input)

Overall, I built a reliable RAG system used by the AI team internally for experimentation and prototype development.

---

## 📥 Installation

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Start Ollama Server

```
ollama run qwen2.5
```

### 3. Run the RAG Pipeline

```
python rag_pipeline.py
```

---

## 📌 Future Enhancements

* Support for multilingual embeddings
* Use of persistent vector DBs like Qdrant or Chroma
* Chunking optimization for large PDFs
* Response ranking (relevance scoring)

---

## 🤝 Acknowledgements

Special thanks to the **AI Team HR Team CEO at Ailaysa, Taramani**, for guidance, review, and collaboration during the project.

---
