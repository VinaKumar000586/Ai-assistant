🤖 AI PDF Smart Search Assistant
"Chat with your documents like never before."
An AI‑powered conversational assistant that can understand, search, and answer questions from your own PDF documents using LangChain, Google Gemini (Generative AI), and Chroma Vector Store.
If the information isn’t in your PDFs, it will automatically search the web for the most relevant answers.

🌟 Key Highlights
📂 Multi‑PDF Knowledge Base – Read and index multiple PDFs for intelligent search.

🧠 Vector Database Integration – Efficient semantic search using Chroma.

🗣 Conversational Memory – Maintains chat context for coherent multi‑turn conversations.

🌐 Web Search Fallback – Integrates Tavily API when answers are missing in PDFs.

🎯 Structured, Detailed Responses – Clear explanations for both technical and non‑technical users.

⚡ Deploy‑Ready – Works locally and on cloud platforms like Streamlit Cloud or Render.

🛠 Architecture Overview
sql
Copy
Edit
                ┌────────────────────┐
                │   User Question    │
                └─────────┬──────────┘
                          │
                 ┌────────▼─────────┐
                 │  SmartSearch AI  │
                 └───────┬──────────┘
                         │
        ┌────────────────┼─────────────────┐
        │                                   │
┌───────▼────────┐                ┌────────▼───────┐
│ Local PDF Data │                │  Web Search    │
│ (ChromaDB)     │                │  (Tavily API)  │
└────────────────┘                └────────────────┘
                         │
                ┌────────▼─────────┐
                │  Google Gemini   │
                │  LLM Processing  │
                └──────────────────┘
📂 Project Structure
bash
Copy
Edit
├── app.py                 # Web UI entry point (Streamlit / Render)
├── mldl_chatboat.py       # Core AI logic
├── pdfs/                  # Store your PDF files here
├── Chroma_db/              # Vector DB storage
├── requirements.txt       # Project dependencies
├── .env                   # API keys
└── README.md              # Documentation
🚀 Setup & Installation
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/YourUsername/ai-pdf-assistant.git
cd ai-pdf-assistant
2️⃣ Create a virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Set up environment variables
Create a .env file:

ini
Copy
Edit
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
5️⃣ Add your PDFs
Place your PDF files in the pdfs/ folder:

Copy
Edit
pdfs/
├── book1.pdf
├── research.pdf
└── notes.pdf
6️⃣ Run the application

bash
Copy
Edit
python mldl_chatboat.py
Or for web deployment:

bash
Copy
Edit
streamlit run app.py
💡 Example Interaction
Q: "Summarize Chapter 4 from my uploaded AI textbook."
A:

Chapter 4 focuses on Neural Network Architectures, covering:

Differences between Feedforward and Recurrent Networks

Real‑world applications in NLP and Computer Vision
(Answer generated using PDF knowledge base + LLM reasoning.)

🧠 Tech Stack
LangChain – AI orchestration

Google Generative AI (Gemini) – LLM

ChromaDB – Vector database

Tavily API – Web search

PyPDFLoader – PDF processing
