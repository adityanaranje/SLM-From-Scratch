# 🧠 NanoChat — A Small Conversational Language Model (From Scratch)

NanoChat is a **decoder-only, causal language model** built entirely from scratch in **PyTorch**, following a NanoGPT-style architecture.  
The model is trained using **next-token prediction** and demonstrates how conversational behavior can emerge from **text completion**, rather than explicit dialogue reasoning.

This project is intended for **educational and research purposes**, focusing on understanding how Transformer-based language models work internally.

---

## 🚀 Project Overview

NanoChat implements a **GPT-style decoder-only Transformer** trained autoregressively:

P(xₜ | x₁, x₂, …, xₜ₋₁)

Although the model is exposed through a chat-style interface, it fundamentally performs **text completion**. Conversational behavior arises from:
- Role-based prompt formatting (`<user>`, `<assistant>`)
- Conversational datasets
- Application-level decoding constraints

The model itself does not natively understand dialogue turns.

---

## 🏗️ Architecture

**Model Type:** Decoder-only Transformer (Causal Language Model)

| Parameter | Value |
|--------|------|
| Layers | 4 |
| Attention Heads | 4 |
| Embedding Size | 512 |
| Context Length | 256 tokens |
| Vocabulary | 8,000 (BPE) |
| Parameters | ~60M |

---

## 🧠 Completion-Based, Not a Native Chatbot

NanoChat predicts the **next most likely token** in a sequence.  
It does **not** inherently:
- Decide when a reply ends
- Understand user intent
- Enforce one-turn responses

The Streamlit application enforces these behaviors externally by:
- Stopping generation at `<eos>` or `<user>`
- Returning only the assistant portion of the generated text

---

## 📊 Training Data

- **TinyStories** — grammar, structure, reasoning  
- **PersonaChat** — conversational patterns

### Training Format

```
<user> {input}
<assistant> {response}
<eos>
```

---

## 🧪 Training Objective

- Causal language modeling (next-token prediction)
- Cross-entropy loss
- No instruction tuning
- No RLHF

---

## 🖥️ Application

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📂 Repository Structure

```
.
├── app.py
├── model.py
├── requirements.txt
├── Data/
│   └── tokenizer.json
├── Model/
│   └── slm_tinystories_personachat.pt
└── Notebooks/
```

---

## ⚠️ Limitations

- Completion-based generation, not intent-aware dialogue
- No instruction tuning or safety alignment
- Limited knowledge from training data only
- Small model capacity (~60M parameters)
- Context window limited to 256 tokens

This model is intended strictly for **learning and experimentation**.

---

## 🛣️ Next Work / Roadmap

This project is an ongoing learning effort. The following steps outline planned extensions to evolve the model from a completion-based SLM into a more complete language system.

### 1️⃣ Model Evaluation & Metrics
- Implement **perplexity** evaluation on a held-out validation set
- Evaluate conversational quality using **BLEU / ROUGE** on PersonaChat-style responses
- Add lightweight **human evaluation** for coherence and relevance

### 2️⃣ Decoding & Inference Improvements
- Enforce strict reply boundaries (stop generation on `<user>` or `<eos>`)
- Tune sampling strategies (temperature, top-k)
- Add repetition and length penalties to reduce rambling

### 3️⃣ Retrieval-Augmented Generation (RAG)
- Integrate a lightweight retrieval pipeline using:
  - Sentence embeddings
  - FAISS for similarity search
- Ground responses using external context without retraining the model
- Evaluate factual consistency before and after retrieval

### 4️⃣ Instruction-Style Fine-Tuning
- Introduce instruction–response examples to improve command following
- Convert the model from pure completion behavior to instruction-aware generation
- Measure improvements using controlled prompts

### 5️⃣ Deployment & Scaling (In Progress)
- ✅ **Streamlit app deployed on Streamlit Cloud**
- Improve startup time and inference latency
- Add caching and resource-aware optimizations
- Optional: expose a minimal inference API (FastAPI)

### 6️⃣ Benchmarking & Analysis
- Compare performance against small pretrained baselines (e.g., GPT-2 small)
- Analyze trade-offs between model size, quality, and training time

The focus remains on **evaluation, system design, and real-world usability**, rather than scaling model size.


---

## 📄 License

MIT License.

