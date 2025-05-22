import os
import streamlit as st
import pickle
import faiss
import numpy as np
from fastembed import TextEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Workaround for OpenMP duplicate library issue (safe for dev)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --------- LLM SETUP (TinyLlama, CPU) ---------
@st.cache_resource
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        device=-1  # CPU
    )
    return pipe

llm = load_llm()

# --------- FAISS & EMBEDDING SETUP ---------
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index("faiss_index.bin")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

@st.cache_resource
def get_embed_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

embed_model = get_embed_model()

def get_embedding(text):
    return np.array(list(embed_model.embed([text]))[0], dtype="float32")

def search_properties(query, top_k=5):
    index, metadata = load_index_and_metadata()
    emb = get_embedding(query).reshape(1, -1)
    D, I = index.search(emb, top_k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

def generate_hf_answer(query, retrieved_properties):
    # Include property links in the LLM context, and ask it to respond conversationally.
    context = "\n".join(
        f"{i+1}. [{prop['title']}]({prop.get('link','')}) | Location: {prop['location']} | Price: {prop['price']} | "
        f"Area: {prop.get('area','N/A')} | Type: {prop.get('property_type','N/A')} | Posted: {prop.get('posted_date','N/A')}"
        for i, prop in enumerate(retrieved_properties)
    )
    prompt = (
        f"You are a friendly Jaipur real estate assistant. The user asked: '{query}'.\n"
        f"Here are the most relevant property listings (with links):\n{context}\n\n"
        f"Reply conversationally and helpfully, referencing these listings (with links). "
        f"Don't invent properties not in the list. If you can't answer, say so politely."
    )
    output = llm(prompt)
    return output[0]["generated_text"][len(prompt):].strip()

# --------- Streamlit Conversational UI ---------
st.title("🏠 Jaipur Property Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about Jaipur properties...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Searching and thinking..."):
        results = search_properties(user_input)
        if not results:
            answer = "Sorry, I couldn't find relevant properties."
        else:
            answer = generate_hf_answer(user_input, results)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        # Also show a card list with clickable links
        if results:
            st.markdown("#### Matching Properties:")
            for i, prop in enumerate(results, 1):
                link = prop.get("link", "")
                st.markdown(
                    f"**{i}. [{prop['title']}]({link})**  \n"
                    f"- Location: {prop['location'] or 'N/A'}  \n"
                    f"- Area: {prop.get('area', 'N/A')}  \n"
                    f"- Price: {prop['price'] or 'N/A'}  \n"
                    f"- Type: {prop.get('property_type','N/A')}  \n"
                    f"- Posted: {prop.get('posted_date','N/A')}  \n"
                    f"- Source: {prop['source']}"
                )

if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.experimental_rerun()