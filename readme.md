# English to Tamil Transformer Translator

This is a Streamlit-powered web app that performs English to Tamil translation using a custom **vanilla Transformer** built from scratch with PyTorch.

Built entirely without using Hugging Face, this project demonstrates a clean, from-scratch implementation of the Transformer architecture (encoder-decoder model with attention), tokenization logic, training infrastructure, and deployment flow.

---

## 🧠 Features

- Custom Transformer implementation (no prebuilt libraries)
- Sequence-to-sequence translation: English → Tamil
- Custom tokenizer (SimpleTokenizer) with word-level vocab
- Model inference using trained `.pth` weights
- Streamlit frontend for live translation demo
- Deployed to Streamlit Cloud

---

## 🚀 Live Demo

[👉 Click to try the app](https://englishtotamil.streamlit.app/)

---

## 🛠 Tech Stack

- Python
- PyTorch (custom Transformer architecture)
- Streamlit (for deployment)
- Pickle (tokenizer persistence)
- Basic file I/O (no database, no nonsense)

---

## 🧪 Model Training

The model was trained using:

- 2-layer encoder and decoder
- `d_model = 128`
- `ff_hidden_dim = 512`
- `n_heads = 8`
- `vocab_size ≈ 185k (src), 464k (tgt)`
- Cross-entropy loss

Due to computational constraints, training was limited to **10 epochs**. The model currently outputs simple translations, and future training (or pretraining on subword-level tokens) can improve accuracy significantly.

---

## Project Structure

english_tamil_transformer/
├── streamlit_app.py # Main Streamlit app
├── transformer_model.py # Full Transformer implementation
├── tokenizer.py # SimpleTokenizer class
├── transformer_en_ta.pth # Trained model weights
├── src_tokenizer.pkl # Tokenizer for English (source)
├── tgt_tokenizer.pkl # Tokenizer for Tamil (target)
├── requirements.txt # Dependencies
└── README.md # This file

---

## Why I Built This

This project was developed to:

- Understand and implement the Transformer architecture from scratch
- Explore bilingual translation for low-resource languages
- Practice full ML project lifecycle: data → model → UI → deployment
- Demonstrate core deep learning and deployment skills

---

## Limitations

- Only trained for 10 epochs on limited hardware
- Word-level tokenizer; no subword/BPE
- Limited corpus and vocabulary
- `<unk>` may occur frequently in output (which is a rite of passage)

---
