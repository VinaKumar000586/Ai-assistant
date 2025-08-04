import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Changed from Chroma to FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER_PATH = os.path.join(BASE_DIR, "pdfs")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")  # Changed from Chroma_db

class DocumentLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        
    def load(self):
        loader = DirectoryLoader(
            path=self.folder_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        return loader.load()

class TextSplitter:
    def __init__(self, chunk_size=2000, chunk_overlap=300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split(self, documents):
        text_split = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_split.split_documents(documents)

class VectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_KEY
        )

    def create_and_persist(self):
        vs = FAISS.from_documents(  # Changed to FAISS
            self.documents,
            self.embedding
        )
        vs.save_local(FAISS_INDEX_PATH)  # Save FAISS index
        return vs
        
    @staticmethod
    def load():
        return FAISS.load_local(  # Changed to FAISS
            FAISS_INDEX_PATH,
            GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_KEY
            ),
            allow_dangerous_deserialization=True
        )

# Build vector store
print("[INFO] Building vector store from PDFs...")
docs = DocumentLoader(PDF_FOLDER_PATH).load()
print(f"[INFO] Loaded {len(docs)} documents from {PDF_FOLDER_PATH}")

split_docs = TextSplitter().split(docs)
print(f"[INFO] Split into {len(split_docs)} chunks.")

VECTOR_STORE = FAISS.from_documents(  # Changed to FAISS
    split_docs,
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_KEY
    )
)
print("[INFO] Vector store ready.")

class SmartSearch:
    def __init__(self, query, memory):
        self.query = query
        self.vector_store = VECTOR_STORE
        self.memory = memory
    
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            max_output_tokens=2064,     
            api_key=GOOGLE_KEY
        )
        
        self.web_search = TavilySearch(k=5, tavily_api_key=TAVILY_KEY)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that provides **detailed, well-structured, and thorough explanations**.

Guidelines:
- Answer in multiple paragraphs when appropriate.
- Use examples, comparisons, and real-world applications.
- If the topic is technical, explain both theory and practical use cases.
- Maintain accuracy while ensuring clarity for beginners and advanced users.
- Avoid one-line or overly brief answers unless the question is trivial.

{extra_instructions}

Context:
{context}

Question:
{question}
""")

    # [Rest of your SmartSearch class methods remain exactly the same]
    # ... (keep all existing methods unchanged)

if __name__ == "__main__":
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    while True:
        query = input("\nAsk me something (or type 'exit'): ")
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        search_tool = SmartSearch(query, memory)
        answer = search_tool.search()

        print(f"\n💡 Answer:\n{answer}\n")
