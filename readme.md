# Jaipur Real Estate Conversational Bot

This project is a conversational chatbot for Jaipur property search. It scrapes property listings from [Magicbricks](https://www.magicbricks.com/) and [Housing.com](https://housing.com/), builds a semantic search index using [fastembed](https://github.com/huggingface/fastembed) and [FAISS](https://github.com/facebookresearch/faiss), and allows users to chat with an LLM (TinyLlama) about available properties via a Streamlit web app.

## Features

- Scrapes property details (title, location, price, area, bathrooms, property type, posted date, and property link) from Magicbricks and Housing.com.
- Stores listings in a semantic search index for fast and relevant retrieval.
- Users can chat with a friendly AI assistant about Jaipur properties.
- Bot responds conversationally and includes property links in responses.

---

## Setup

### 1. Clone the repo (or copy the files)

```
git clone https://github.com/YOUR-USERNAME/Real_Estate_scrapping.git

```

### 2. Install dependencies

It's recommended to use a virtual environment:
```sh
cd Real_Estate_scrapping
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Download LLM weights (TinyLlama)

The first run will automatically download the TinyLlama model.  
For slow connections, you can manually download [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) from Hugging Face and place it in a local directory.

### 4. Run the scraper

Scrape the latest property listings and build the FAISS index (this will create `metadata.pkl` and `faiss_index.bin`):

```sh
python scrape_and_index.py
```

### 5. Launch the Streamlit chatbot

```sh
streamlit run app.py --logger.level=error
```

Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

- Type your questions about Jaipur properties in the chat input (e.g., "3 BHK in Malviya Nagar under 80 lakhs").
- The assistant will reply with relevant listings and clickable property links.
- Clear the conversation anytime with the "Clear Conversation" button.

---

## Notes

- The scraper uses HTML parsing and may break if the target websites change their structure.
- The default LLM is TinyLlama 1.1B; you can swap for any HuggingFace-compatible conversational model.
- If you plan to deploy or use at scale, respect the scraping policies of source sites.

---

## Troubleshooting

- If you see errors about `KMP_DUPLICATE_LIB_OK`, it's set automatically in `app.py` for local runs.
- For download timeouts, ensure your internet connection is stable or download model files manually.

---

## License

MIT (or specify your license here)

---

## Acknowledgements

- [Magicbricks](https://magicbricks.com) and [Housing.com](https://housing.com) for property data.
- [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search.
- [TinyLlama](https://huggingface.co/TinyLlama) for the open conversational model.