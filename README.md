# 🎉 El7amalawy 🎉 V1.2

A Streamlit chatbot with RAG that lets you upload multiple PDF files, search across them, and chat with your documents using Groq free models.

## Features

- Upload multiple PDFs
- RAG pipeline for searching inside the PDFs
- Groq API key input from the sidebar
- Free Groq model selection
- Chat history in the app
- Top `K=3` retrieved chunks
- Similarity percentage shown for each retrieved chunk

## Tech Stack

- Streamlit
- Groq API
- LangChain community tools
- Chroma vector store
- PyPDF
- Local hash-based embeddings for free retrieval

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
└── RAG/
```

## How To Run

1. Clone the repository:

```bash
git clone https://github.com/El7amalawy/El7amalawy-chatbot-with-RAG.git
cd El7amalawy-chatbot-with-RAG
```

2. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
python -m streamlit run app.py
```

4. Open the app in your browser:

```text
http://localhost:8501
```

## How To Use

1. Open the app.
2. Paste your Groq API key in the sidebar.
3. If you do not have one, get it from:
   `https://console.groq.com/keys`
4. Upload one or more PDF files.
5. Click `Process PDFs`.
6. Ask questions in the chat input.
7. Review the top 3 retrieved chunks and their similarity scores.

## Notes

- This app uses Groq for generation only.
- Retrieval uses a free local embedding approach, so the user only needs a Groq API key.
- If the answer is not found in the uploaded PDFs, the chatbot says it does not have enough information.

## Repository

GitHub repo:
`https://github.com/El7amalawy/El7amalawy-chatbot-with-RAG`
