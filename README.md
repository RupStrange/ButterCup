# 🧈 ButterCup

### Your AI companion for understanding YouTube videos.

**Paste a YouTube link. Get the key ideas. Ask anything.**

ButterCup turns long YouTube videos into something you can actually interact with.

🎥 **Watch less. Understand more.**

---

## ✨ What can ButterCup do?

### 📝 Understand any video

Drop in a YouTube URL and ButterCup creates a clear, structured summary of the video.

No need to sit through an hour-long video just to find three useful minutes.

### 💬 Ask the video anything

Ask questions naturally:

> "What was the main argument?"

> "What examples did the speaker give?"

> "What are the disadvantages mentioned?"

ButterCup answers using the video's content whenever the answer is actually there.

### 🌐 What if the video doesn't know?

This is where ButterCup gets interesting.

If your question goes beyond what the video discusses, ButterCup can search the web and bring that information into the conversation.

So instead of pretending the video contains an answer, it can tell the difference between:

**"The video says this."**

and

**"The video doesn't cover this — here's what I found elsewhere."**

### 🔎 Know where your answer came from

Every answer makes its source clear:

🟢 **Video**
🌐 **Web**
🟡 **Video + Web**

You always know whether you're learning from the video or from outside sources.

### 🧠 Keep the conversation going

You don't have to repeat yourself.

Ask a follow-up question and ButterCup remembers the recent conversation to keep the discussion flowing.

---

## 🚀 How it works

It's simple from the user's perspective:

```text
        🎥 YouTube Video
              ↓
        📝 Understand it
              ↓
        🧠 Ask questions
              ↓
     ┌────────┴────────┐
     ↓                 ↓
Video has it       Video doesn't
     ↓                 ↓
  🎬 Answer         🌐 Search Web
     └────────┬────────┘
              ↓
        💬 Your Answer
```

Behind the scenes, ButterCup uses **Corrective RAG** to check whether the retrieved video content is actually useful before answering.

---

## 🎯 Why ButterCup?

Most YouTube AI tools stop at:

> **"Here's your summary."**

ButterCup goes one step further:

> **"Let's talk about the video."**

And when the video isn't enough, it can look beyond the video instead of forcing an answer from irrelevant information.

---

## 🛠️ Built With

* 🐍 Python
* ⚡ Streamlit
* 🧠 LangGraph
* 🤖 Groq
* 🔎 FAISS
* 🌐 Tavily
* 🔤 BAAI BGE embeddings

---

## 🚀 Run ButterCup

### 1. Clone

```bash
git clone https://github.com/RupStrange/ButterCup.git
cd ButterCup
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Add your API keys

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
TAVILY_API_KEY = "your-tavily-api-key"
```

### 4. Start ButterCup

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

---

## 🗺️ What's next?

ButterCup is still evolving.

Planned improvements include:

* ⚡ Faster responses
* 🎥 Multiple videos in one conversation
* 💬 Cross-video conversations
* 🧪 Better answer and retrieval evaluation
* 🚀 Further performance improvements

---

## 👨‍💻 Built By

**Sourasish Das**

Built as a hands-on exploration of **Generative AI, RAG, and intelligent AI applications.**

---

### 🧈 ButterCup

**YouTube → Knowledge → Conversation**

> **Don't just watch the video. Talk to it.**
