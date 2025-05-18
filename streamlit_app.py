import streamlit as st
import torch
from transformer_model import Transformer, greedy_decode
from tokenizer import SimpleTokenizer
import pickle

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "transformer_weights.pth"

# Hyperparameters
SRC_VOCAB_SIZE = 185257
TGT_VOCAB_SIZE = 464454
EMB_SIZE = 128
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
MAX_LEN = 20

@st.cache_resource
def load_model():
    model = Transformer(
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE,
        NUM_ENCODER_LAYERS, NHEAD, FFN_HID_DIM, MAX_LEN
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

with open('src_tokenizer.pkl', 'rb') as f:
    src_tokenizer = pickle.load(f)

with open('tgt_tokenizer.pkl', 'rb') as f:
    tgt_tokenizer = pickle.load(f)

model = load_model()

st.title("🌐 English to Tamil Translator")
st.markdown("Using a custom vanilla Transformer model built from scratch")

input_sentence = st.text_input("Enter English sentence:", value="<sos> The book is on the table <eos>")

if st.button("Translate"):
    encoded_input = src_tokenizer.encode(input_sentence)
    src_tensor = torch.tensor([encoded_input]).to(DEVICE)
    
    with torch.no_grad():
        output = greedy_decode(model, input_sentence, src_tokenizer, tgt_tokenizer, device=DEVICE)

    
    st.subheader("📘 Tamil Translation")
    st.write(output)


st.markdown("---")
st.caption("Trained for 10 epochs due to compute limitations. Demo powered by PyTorch + Streamlit.")
