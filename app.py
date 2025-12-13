import streamlit as st
import torch
from tokenizers import Tokenizer
from model import GPT, GPTConfig
import os
import urllib.request

MODEL_URL = "https://github.com/adityanaranje/SLM-From-Scratch/releases/download/Model/slm_tinystories_personachat.pt"
MODEL_PATH = "slm_tinystories_personachat.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

download_model()


st.set_page_config(page_title="NanoGPT-Completion")

@st.cache_resource
def load_model():
    tokenizer = Tokenizer.from_file("Data/tokenizer.json")

    config = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=512,
    )

    model = GPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, tokenizer

model, tokenizer = load_model()

st.title("🧠 Nano-GPT Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:")

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.2, step= 0.1, value=0.8)
max_tokens = st.sidebar.selectbox("Max Tokens",options=[30, 40, 50, 60, 70])
top_k = st.sidebar.number_input("Top-K", min_value=10, max_value=40, step=1, value=30)

if user_input:
    prompt = f"<user> {user_input}\n<assistant>"
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids).unsqueeze(0)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    text = tokenizer.decode(y[0].tolist())
    reply = text.split("<assistant>")[-1].split("<eos>")[0].strip()

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))

for speaker, msg in st.session_state.history:
    st.markdown(f"**{speaker}:** {msg}")




