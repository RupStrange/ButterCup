Absolutely 👍 For a GitHub portfolio project, I'd make it a little more **professional and visually polished** rather than just adding lots of emojis.

Below is a **GitHub-ready `README.md`**. You can copy everything inside the code block directly into your `README.md`.

````markdown
# 🧈 ButterCup — YouTube AI Analyst with Corrective RAG

<p align="center">
  <b>🎥 Turn any YouTube video into a summary and an intelligent ask-anything assistant.</b>
</p>

<p align="center">
  <i>
    ButterCup combines YouTube transcripts, FAISS retrieval, LLM-based relevance grading,
    context refinement, and live web search using a Corrective RAG (CRAG) pipeline.
  </i>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![Groq](https://img.shields.io/badge/Groq-LLM-black)
![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-purple)

</p>

---

## 📌 Overview

**ButterCup** is an AI-powered YouTube analyst that allows users to paste a YouTube URL, generate a structured summary, and ask questions about the video.

Unlike a basic RAG chatbot, ButterCup evaluates whether the retrieved transcript actually contains information relevant to the user's question.

If the retrieved information is:

- 🟢 **Relevant** → use the video context
- 🟡 **Ambiguous** → combine video context with web search
- 🔴 **Irrelevant** → discard the context and search the web

This approach is implemented using **Corrective RAG (CRAG)** with **LangGraph**.

---

## ✨ Features

### 🎬 YouTube Video Analysis

Paste a YouTube URL and ButterCup:

- 🔗 Extracts the video ID
- 🎥 Fetches video metadata
- 📝 Retrieves the transcript
- 🌍 Handles transcript translation/cleanup
- 🧠 Generates a structured summary

---

### 💬 Ask Anything About the Video

Users can ask follow-up questions such as:

> "What was the main argument of the speaker?"

> "What examples were mentioned?"

> "What are the disadvantages discussed in the video?"

Answers are grounded in the video's transcript whenever relevant information is available.

---

### 🧠 Corrective RAG (CRAG)

ButterCup doesn't blindly trust retrieved documents.

Every retrieved transcript chunk is independently evaluated by an LLM for relevance.

This allows the system to detect when retrieval is:

```text
        Useful
          │
          ▼
      CORRECT
          │
          │
   ┌──────┴──────┐
   ▼             ▼
Answer       Refine Context


       Uncertain
          │
          ▼
      AMBIGUOUS
          │
          ▼
     Web Search
          │
          ▼
   Video + Web


      Irrelevant
          │
          ▼
     INCORRECT
          │
          ▼
     Web Search
````

---

### 🌐 Automatic Web Fallback

If the video's transcript doesn't contain enough information to answer the question, ButterCup automatically:

1. 🔄 Rewrites the user's query
2. 🌐 Performs a live Tavily search
3. 📄 Retrieves relevant web information
4. 🧠 Uses the results to generate the answer

This allows the application to distinguish between:

> **"The video doesn't discuss this."**

and

> **"The information exists elsewhere on the web."**

---

### 🔎 Source Transparency

Each answer indicates its information source:

| Indicator | Source      |
| --------- | ----------- |
| 🟢        | Video only  |
| 🌐        | Web only    |
| 🟡        | Video + Web |

---

### 🧠 Conversation Memory

ButterCup maintains recent conversation history so follow-up questions can use previous turns as context.

For example:

```text
User: What is RAG?

Assistant: RAG stands for Retrieval-Augmented Generation...

User: Why did the speaker prefer it?

Assistant: Based on the previous context...
```

---

# ⚙️ How It Works

```text
                         👤 USER QUESTION
                                │
                                ▼
                     ┌────────────────────┐
                     │     🔍 RETRIEVE    │
                     │                    │
                     │  FAISS + MMR Search │
                     └──────────┬─────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │    🧠 EVALUATE DOCS      │
                  │                          │
                  │ LLM relevance scoring    │
                  │       0.0 → 1.0          │
                  └────────────┬─────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        🟢 CORRECT        🟡 AMBIGUOUS       🔴 INCORRECT
              │                │                │
              │                ▼                ▼
              │          🔄 Query Rewrite   🔄 Query Rewrite
              │                │                │
              │                └──────┬─────────┘
              │                       ▼
              │               🌐 TAVILY SEARCH
              │                       │
              └───────────────┬───────┘
                              ▼
                    ┌────────────────────┐
                    │      ✂️ REFINE     │
                    │                    │
                    │ Keep only relevant │
                    │ transcript content │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │     ✨ GENERATE    │
                    │                    │
                    │ History-aware final │
                    │      response      │
                    └────────────────────┘
```

---

## 🔬 CRAG Pipeline

### 1️⃣ Retrieve

The system retrieves the top-k chunks from the video's FAISS vector index.

ButterCup uses **MMR (Maximal Marginal Relevance)** to improve diversity between retrieved chunks rather than relying only on similarity scores.

```text
YouTube Transcript
        │
        ▼
   Text Chunks
        │
        ▼
   Embeddings
        │
        ▼
      FAISS
        │
        ▼
   MMR Retrieval
        │
        ▼
 Top-K Documents
```

---

### 2️⃣ Evaluate

Each retrieved document is independently evaluated by the LLM.

The model produces a relevance score:

```text
0.0 ─────────────────────────────── 1.0
│                                    │
Irrelevant                        Relevant
```

---

### 3️⃣ Decide

Two configurable thresholds determine the CRAG decision.

```text
Score > Upper Threshold
        │
        ▼
    🟢 CORRECT


Lower < Score < Upper
        │
        ▼
    🟡 AMBIGUOUS


Score < Lower Threshold
        │
        ▼
    🔴 INCORRECT
```

---

### 4️⃣ Refine

Relevant transcript context is split into individual sentences.

The LLM filters out sentences that aren't useful for answering the question.

```text
Retrieved Context
       │
       ▼
Split into sentences
       │
       ▼
LLM relevance filtering
       │
       ▼
Relevant sentences only
```

This keeps unnecessary transcript information out of the final prompt.

---

### 5️⃣ Generate

The final response is generated using:

* 📄 Refined context
* 🌐 Web results when required
* 🧠 Conversation history
* ❓ User's current question

---

# 🏗️ Architecture

ButterCup originally started as a single **~460-line Streamlit script** containing:

* UI code
* Prompts
* LLM calls
* Retrieval logic
* Progress indicators
* Application logic

The project has since been refactored into separate modules based on responsibility.

### 🎯 Separation of Concerns

Only:

```text
app.py
```

imports Streamlit.

The CRAG and supporting components are plain Python modules.

This makes the system easier to:

* 🧪 Test
* ♻️ Reuse
* 💻 Run from a CLI
* 🌐 Wrap with an API
* 🔧 Modify independently

---

# 🛠️ Tech Stack

| Layer                | Technology               |
| -------------------- | ------------------------ |
| 🎨 UI                | Streamlit                |
| 🤖 LLM               | Groq + `langchain-groq`  |
| 🔄 Orchestration     | LangGraph                |
| 🗃️ Vector Store     | FAISS                    |
| 🧮 Embeddings        | Sentence Transformers    |
| 🔤 Embedding Model   | `BAAI/bge-small-en-v1.5` |
| 🌐 Web Search        | Tavily                   |
| 🎥 Transcript        | `youtube-transcript-api` |
| 📦 Structured Output | Pydantic v2              |
| 🧠 RAG               | Corrective RAG (CRAG)    |

---

# 🚀 Getting Started

## 📋 Prerequisites

Make sure you have:

* 🐍 Python **3.10+**
* 🔑 Groq API key
* 🔑 Tavily API key

---

## 📥 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/ButterCup.git
cd ButterCup
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Configure API Keys

Copy the example secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then add your keys:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
TAVILY_API_KEY = "your-tavily-api-key-here"
```

> ⚠️ **Never commit your actual API keys.**

The following files are already ignored by Git:

```text
.streamlit/secrets.toml
buttercup/secrets.toml
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will start locally and Streamlit will provide the URL in the terminal.

---

# ⚙️ Configuration

Configuration is centralized in:

```text
buttercup/config.py
```

### CRAG Configuration

| Setting                | Default | Description                                      |
| ---------------------- | ------: | ------------------------------------------------ |
| `CRAG_UPPER_THRESHOLD` |   `0.7` | Score above this can produce a `CORRECT` verdict |
| `CRAG_LOWER_THRESHOLD` |   `0.3` | Scores below this are treated as irrelevant      |
| `retriever_k`          |     `4` | Number of chunks retrieved per question          |
| `memory_max_messages`  |    `12` | Number of previous messages retained             |

---

# 📁 Project Structure

```text
buttercup_project/
│
├── 📄 app.py
│   └── Streamlit UI only
│
├── 📄 requirements.txt
│
├── 📁 .streamlit/
│   └── secrets.toml.example
│
└── 📁 buttercup/
    │
    ├── ⚙️ config.py
    │   └── Tunable constants + secret lookup
    │
    ├── 📦 schemas.py
    │   └── Pydantic models + CRAGState
    │
    ├── 🎥 youtube_service.py
    │   └── URL parsing + metadata + transcript fetching
    │
    ├── 📝 transcript_processor.py
    │   └── Translation + transcript cleanup
    │
    ├── 🧠 summarizer.py
    │   └── Fact extraction + summary generation
    │
    ├── 🗃️ vectorstore_builder.py
    │   └── FAISS retriever construction
    │
    ├── 💾 memory_store.py
    │   └── Conversation memory wrapper
    │
    └── 📁 crag/
        │
        ├── 🔍 grading.py
        │   └── Retrieval + relevance evaluation
        │
        ├── ✂️ refine.py
        │   └── Sentence decomposition + filtering
        │
        ├── 🌐 web_search.py
        │   └── Query rewriting + Tavily search
        │
        ├── ✨ generate.py
        │   └── History-aware answer generation
        │
        └── 🔗 graph.py
            └── Compiled LangGraph StateGraph
```

---

# 🧠 Design Notes

## ⏱️ Latency Trade-off

CRAG evaluates every retrieved chunk using a separate LLM call and then performs additional filtering during refinement.

With:

```text
retriever_k = 4
```

a single question can require approximately:

```text
6–10+ LLM calls
```

This is acceptable for a demonstration, but production deployments could reduce latency through:

* ⚡ Batch grading
* 🔄 Parallel execution
* 📦 Concatenated document grading
* 🚀 Reduced refinement calls

For example:

```python
chain.batch(...)
```

could be used to process multiple grading requests together.

---

## 📦 Structured Output

ButterCup uses:

```python
with_structured_output(...)
```

to obtain predictable structured responses from the LLM.

This requires a Groq model that supports tool/function calling.

---

## 🌐 Tavily Integration

The current implementation uses:

```text
TavilySearchResults
```

from the LangChain community integration.

The project can be migrated to the newer:

```text
langchain-tavily
```

integration by updating the web-search implementation.

---

# 🗺️ Roadmap

* [ ] ⚡ Batch CRAG grading calls
* [ ] 🔄 Parallelize independent CRAG operations
* [ ] 🌐 Migrate to `langchain-tavily`
* [ ] 🧪 Add automated tests
* [ ] 🎥 Support multiple videos in one session
* [ ] 💬 Enable cross-video conversations
* [ ] 🚀 Further reduce CRAG latency
* [ ] 📊 Add evaluation metrics for retrieval quality

---

# 🔮 Future Improvements

Potential future improvements include:

### 📊 RAG Evaluation

Add metrics such as:

* Retrieval precision
* Retrieval recall
* Answer faithfulness
* Context relevance
* Answer relevance

### ⚡ Performance Optimization

Reduce the number of LLM calls through:

* Batched grading
* Parallel execution
* Smarter retrieval
* Cached evaluations

### 🎥 Multi-Video Knowledge

Allow users to upload or analyze multiple videos and ask questions across all of them.

Example:

```text
Video A ──┐
Video B ──┼──► Shared Vector Store ──► CRAG ──► Answer
Video C ──┘
```

---

# ⭐ Why ButterCup?

ButterCup goes beyond a traditional YouTube summarizer.

It combines:

```text
🎥 YouTube Video
       ↓
📝 Transcript
       ↓
🧮 Embeddings
       ↓
🗃️ FAISS Retrieval
       ↓
🧠 LLM Relevance Grading
       ↓
✂️ Context Refinement
       ↓
🌐 Web Fallback
       ↓
💬 Conversational Answer
```

The key idea is simple:

> **Use the video's knowledge when it is relevant. Search the web when it isn't.**

---

# 🧈 Built With

<p align="center">

**LangGraph • Groq • FAISS • Tavily • Streamlit • Python**

</p>

<p align="center">
  🎥 Video → 🧠 Knowledge → 💬 Conversation
</p>

---

## 👨‍💻 Author

**Sourashish Das**

Built as an exploration of **Generative AI, RAG, LangGraph orchestration, LLM-based evaluation, and agentic workflows.**

---

````

### A couple of things I'd recommend before you push it

**1. Add a screenshot of the actual ButterCup UI.**  
This is probably the single biggest improvement for a GitHub portfolio README. Put it directly below the Overview:

```markdown
## 🖥️ Demo

![ButterCup Demo](assets/demo.png)
````

and keep the screenshot at:

```text
ButterCup/
├── assets/
│   └── demo.png
├── app.py
└── README.md
```

**2. If you have a deployed version**, add this near the top:

```markdown
## 🚀 Live Demo

👉 **[Try ButterCup](YOUR_DEPLOYED_URL)**
```

**3. Don't use the `<your-username>` URL** if you're using your actual repository. Replace it with your GitHub repo URL.
