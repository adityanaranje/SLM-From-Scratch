# 🧠 Nano-GPT Chatbot (Built from Scratch)

This project implements a chatbot using a custom-built Generative Pre-trained Transformer (GPT) model, written entirely from scratch in PyTorch. It demonstrates the core principles of modern Large Language Models (LLMs) on a smaller scale, suitable for educational purposes and understanding the internal mechanics of Transformers.

## 🚀 Project Overview

The core of this project is a decoder-only Transformer model designed to generate text based on user input. Unlike using pre-made libraries like Hugging Face's `transformers` for the model architecture, this project defines the entire model structure layer-by-layer: attention mechanisms, feed-forward networks, and embeddings.

The application is wrapped in a Streamlit UI for easy interaction.

## 🏗️ Architecture

The model architecture is defined in `model.py` and follows the standard GPT (Generative Pre-trained Transformer) design. 

### Transformer Type: Decoder-Only
This model utilizes a **Decoder-only** Transformer architecture. 
- **Why Decoder-only?**: Unlike Encoder-only models (like BERT) which are good for understanding/classification, or Encoder-Decoder models (like T5) useful for translation, **Decoder-only** models are specialized for **generative tasks**. They generate text autoregressively, meaning they predict the next token based solely on previous tokens.

Here is a detailed breakdown of the components:

### Architecture Diagram

```mermaid
graph TD
    Input[Input Token IDs] --> TokEmb[Token Embeddings]
    Input --> PosEmb[Positional Embeddings]
    TokEmb --> Sum((+))
    PosEmb --> Sum
    Sum --> BlockStart[Transformer Block 1]
    
    subgraph Transformer Block
        direction TB
        BlockInput[Input] --> LN1[Layer Norm 1]
        LN1 --> Attn[Multi-Head Causal Self-Attention]
        Attn --> Add1((+))
        BlockInput --> Add1
        Add1 --> LN2[Layer Norm 2]
        LN2 --> MLP[Feed Forward (MLP)]
        MLP --> Add2((+))
        Add1 --> Add2
    end
    
    BlockStart -.-> BlockEnd[Transformer Block N]
    BlockEnd --> FinalLN[Final Layer Norm]
    FinalLN --> Head[LM Head (Linear)]
    Head --> Output[Output Logits]
```

### 1. Configuration (`GPTConfig`)
The model behaves according to a set of hyperparameters:
- **`vocab_size`**: The size of the vocabulary (defined by the tokenizer).
- **`block_size`**: The maximum sequence length (context window) the model can handle (set to 256).
- **`n_layer`**: The number of Transformer blocks stacked on top of each other (4 layers).
- **`n_head`**: The number of attention heads in the multi-head attention mechanism (4 heads).
- **`n_embd`**: The dimensionality of the embeddings (512 dimensions).

### 2. Causal Self-Attention (`CausalSelfAttention`)
This is the heart of the Transformer. It allows the model to weigh the importance of different words in the sequence when generating the next token.
- **Query, Key, Value**: Linear projections of the input.
- **Scaled Dot-Product Attention**: Computes attention scores.
- **Causal Mask**: A triangular mask ensures that a specific token can only attend to previous tokens (and itself), preventing it from "seeing the future." This is crucial for autoregressive text generation.
- **Multi-Head**: The attention mechanism runs in parallel `n_head` times, allowing the model to capture different types of relationships in the data.

### 3. Transformer Block (`Block`)
Each block consists of two main sub-layers:
1.  **Multi-Head Causal Self-Attention**: (described above).
2.  **Feed-Forward Network (MLP)**: A simple neural network that processes information individually for each position. It expands the dimension by 4x and then projects it back.
    - Structure: Linear -> GELU (Activation) -> Linear.

Both sub-layers use **Layer Normalization** (`LayerNorm`) before the operation (Pre-LN) and add a **Residual Connection** (skip connection) after.

### 4. The GPT Model (`GPT`)
The main class assembles everything:
- **Token Embeddings (`tok_emb`)**: Converts token IDs into vectors of size `n_embd`.
- **Positional Embeddings (`pos_emb`)**: Adds information about the position of each token in the sequence (since attention is permutation invariant).
- **Transformer Blocks**: A stack of `n_layer` blocks.
- **Final Layer Norm**: Normalizes the output of the last block.
- **Language Head (`head`)**: A linear layer that projects the final embedding back to the vocabulary size to predict the next token logits.

## 📂 File Structure

- **`app.py`**: The Streamlit application entry point. Handles user interface, input processing, and inference.
- **`model.py`**: Contains the complete PyTorch definition of the GPT architecture.
- **`requirements.txt`**: List of Python dependencies.
- **`Data/`**: Contains the tokenizer file (`tokenizer.json`).
- **`Modal/`**: Contains the pre-trained model checkpoint (`slm_tinystories_personachat.pt`).

## 🛠️ Installation

1.  **Clone the repository** (if applicable).
2.  **Create a virtual environment** (optional but recommended).
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🧪 Development Pipeline

The project follows a structured development pipeline documented in the `Notebooks/` directory. If you wish to retrain the model or understand the data processing steps, refer to these notebooks:

### 1. Datasets (`Notebooks/DataCollection.ipynb`)
The model is trained on a combination of two datasets sourced from Hugging Face:
-   **[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)**: A collection of short stories containing vocabulary limited to that of a 3-4 year old. This helps the model learn grammar and story structure efficiently.
-   **[awsaf49/persona-chat](https://huggingface.co/datasets/awsaf49/persona-chat)**: A dataset consisting of conversations between two participants who have specific "personas". This provides the model with conversational capabilities.

**Data Formatting**:
The datasets are processed and combined into a specific format to train the model as a chatbot:
-   **TinyStories**: Formatted as a user request ("Tell me a short story") followed by the story content.
-   **PersonaChat**: Formatted as a dialogue between `<user>` and `<assistant>`.
-   **Structure**: `<user> {input}\n<assistant> {response}\n<eos>`

### 2. Data Processing & Tokenization (`Notebooks/Token_and_Torch.ipynb`)
This notebook handles the initial preparation of the dataset:
-   **Tokenizer Training**: Trains a Byte-Pair Encoding (BPE) tokenizer on the raw text data (`train.txt`, `val.txt`).
-   **Vocabulary Generation**: Saves the tokenizer configuration to `Data/tokenizer.json` (vocab size: 8000).
-   **Tensor Creation**: Converts the raw text into PyTorch tensors (`train_ids.pt`, `val_ids.pt`) for efficient loading during training.

### 3. Model Training (`Notebooks/ModelTraining.ipynb`)
This notebook contains the complete training loop:
-   **Model Definition**: Re-defines the `GPT`, `Block`, and `CausalSelfAttention` classes (mirrors `model.py`).
-   **Configuration**: Sets hyperparameters (Block size: 128, Layers: 4, Heads: 4, Embedding dim: 512). *Note: These parameters must match the inference config in `app.py`*.
-   **Training Loop**: Trains the model using the AdamW optimizer with a learning rate of 3e-4.
-   **Checkpointing**: Saves the final trained weights to `slm_tinystories_personachat.pt`.

## ▶️ Usage

To run the chatbot application:

```bash
streamlit run app.py
```

This will launch a web interface where you can chat with the model. You can adjust generation parameters like **Temperature**, **Max Tokens**, and **Top-K** directly from the sidebar.

## 🧠 Inference Flow

1.  **Input**: User types a message.
2.  **Tokenization**: The message is converted into token IDs using the custom tokenizer.
3.  **Model Pass**: The IDs are fed into the `GPT` model.
4.  **Generation**: The model predicts the next token iteratively (autoregressively) until `<eos>` is generated or the max token limit is reached.
5.  **Decoding**: The generated token IDs are converted back into text and displayed.
