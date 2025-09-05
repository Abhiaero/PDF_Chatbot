# PDF Chatbot with Gemini Flash 1.5

A Streamlit application that allows users to upload PDF documents and chat with them using Google's Gemini Flash 1.5 model.

## Features

- 📄 PDF upload and text extraction
- 🔍 Semantic search using FAISS vector database
- 💬 Conversational AI with chat history
- 📑 Source citations with page numbers

## Setup


📂 Project Structure
pdf-chatbot/
├── app.py                 # Main Streamlit app
├── modules/               # Core modules (embedding, retrieval, etc.)
├── utils/                 # Utility functions
├── storage/               # Local storage for processed data
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .env                   # Environment variables (not committed)

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/Abhiaero/PDF_Chatbot.git
cd PDF_Chatbot

2. Create and activate a virtual environment

Windows (PowerShell):

python -m venv venv
venv\Scripts\activate


Linux/Mac (bash):

python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Configure environment variables

Create a .env file in the root folder and add your API keys.
For example, if you’re using OpenAI:

OPENAI_API_KEY=your_api_key_here


⚠️ .env is ignored by Git — each user must create their own.

▶️ Running the App

Run with Streamlit:

streamlit run app.py


Open your browser at:
👉 http://localhost:8501

🛠️ Requirements

Python 3.11+

Streamlit

OpenAI API key (or other LLM provider depending on your setup)

📌 Notes

If you want to reset the chatbot’s knowledge, clear the storage/ folder.

Large binary files should be handled with Git LFS.

Contributions are welcome! Fork the repo and submit a PR.



# How is the conversation history maintained?
Conversation History: Maintained in Streamlit's session state as a list of message dictionaries with role, content, timestamp, and sources. Last 6 messages are used as context for follow-up questions to maintain conversation flow.
Implementation: st.session_state.messages stores entire chat history, and the RAG prompt includes recent conversation context for coherent multi-turn dialogues.

# Limitations:
Current Limitations: Handles only text-based PDFs (not scanned/image PDFs), and may struggle with complex layouts or tables that span multiple pages due to chunking constraints.
Technical Constraints: Limited by Gemini API rate limits and cannot process very large PDFs (>100 pages) efficiently due to local FAISS index size limitations.