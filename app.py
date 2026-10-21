"""
AI Text Summarizer - Streamlit app
B.Tech ECE final-year project, Techno India University (team of 4)

Generates abstractive summaries of news articles using a fine-tuned T5-base model.

The trained model is NOT stored in this repository (it is too large for GitHub).
The app looks for the model in this order:
  1. A local folder named "BTech_Project_Model" next to this file
  2. A Hugging Face Hub model id given in the MODEL_ID environment variable
     (for example: MODEL_ID="your-username/t5-bbc-summarizer")
See the README for how to get or create the model.
"""

import os

import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Text Summarizer", page_icon="🚀")


# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    # 1. Get the directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Local model folder (must match the folder name exactly)
    local_path = os.path.join(current_dir, "BTech_Project_Model")

    # 3. Optional: model hosted on the Hugging Face Hub
    hub_model_id = os.environ.get("MODEL_ID", "").strip()

    if os.path.exists(local_path):
        # Load from local files only
        tokenizer = T5Tokenizer.from_pretrained(local_path, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(local_path, local_files_only=True)
    elif hub_model_id:
        tokenizer = T5Tokenizer.from_pretrained(hub_model_id)
        model = T5ForConditionalGeneration.from_pretrained(hub_model_id)
    else:
        st.error(f"⚠️ Model folder not found at: {local_path}")
        st.info(
            "Place your unzipped model folder named 'BTech_Project_Model' in the same "
            "directory as app.py, or set the MODEL_ID environment variable to a "
            "Hugging Face Hub model id. See the README for details."
        )
        st.stop()

    # Detect hardware (GPU vs CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, device


# --- MAIN UI ---
st.title("🚀 AI Text Summarizer")
st.markdown(
    """
This application uses a **Fine-tuned T5-Base** model to generate abstractive summaries of news articles.
It was trained on the BBC News dataset specifically for this task.
"""
)

# Load the model
try:
    tokenizer, model, device = load_model()
    st.sidebar.success(f"Model loaded on: {device.upper()}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Text input area
text_input = st.text_area(
    "Paste your news article here:",
    height=300,
    placeholder="Enter a long news article to see the AI summarize it...",
)

# Summarize button
if st.button("Generate Summary"):
    if text_input.strip() == "":
        st.warning("Please paste an article first!")
    elif len(text_input.split()) < 30:
        st.warning("The article is too short to summarize. Try a longer text.")
    else:
        with st.spinner("AI is thinking and condensing..."):
            try:
                # 1. Preprocessing (T5 expects a task prefix)
                input_text = "summarize: " + text_input
                inputs = tokenizer(
                    input_text, return_tensors="pt", max_length=512, truncation=True
                ).to(device)

                # 2. Generation with the tuned parameters
                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    max_length=120,  # Allows for full sentences
                    min_length=40,
                    no_repeat_ngram_size=3,
                    length_penalty=1.5,  # Encourages natural flow
                    early_stopping=True,
                )

                # 3. Decoding
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Post-processing so the summary does not stop mid-sentence
                if not summary.endswith("."):
                    summary = summary[: summary.rfind(".") + 1]

                # 4. Display result
                st.subheader("Final Summary")
                st.success(summary)

                # Display metrics (word count reduction)
                original_len = len(text_input.split())
                summary_len = len(summary.split())
                st.info(
                    f"Original: {original_len} words | Summary: {summary_len} words | "
                    f"Reduction: {100 - (summary_len / original_len * 100):.1f}%"
                )

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")

# Footer
st.markdown("---")
st.caption("B.Tech ECE Project | Fine-tuned T5 Transformer")
