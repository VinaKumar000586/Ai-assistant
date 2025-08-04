import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain.memory import ConversationBufferMemory
# Load environment variables
load_dotenv()
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER_PATH = os.path.join(BASE_DIR, "pdfs")  # Folder with PDFs
CHROMA_DB_PATH = os.path.join(BASE_DIR, "Chroma_db")

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
        vs = Chroma.from_documents(
            self.documents,
            self.embedding,
            persist_directory='Chroma_db',
            collection_name='Book_collection'
        )
        vs.persist()
        return vs
        
    @staticmethod
    def load():
        return Chroma(
            persist_directory='Chroma_db',
            collection_name='Book_collection',
            embedding_function=GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_KEY
            )
        )
        
# Build vector store directly in memory (better for deployment)
print("[INFO] Building vector store from PDFs in memory...")
docs = DocumentLoader(PDF_FOLDER_PATH).load()
print(f"[INFO] Loaded {len(docs)} documents from {PDF_FOLDER_PATH}")

split_docs = TextSplitter().split(docs)
print(f"[INFO] Split into {len(split_docs)} chunks.")

VECTOR_STORE = Chroma.from_documents(
    split_docs,
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_KEY
    ),
    collection_name='Book_collection'
)
print("[INFO] Vector store ready in memory.") for better performance 

class SmartSearch:
    def __init__(self, query,memory):
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

        
    # Detect if question is asking for code
    def _needs_code(self, question: str) -> bool:
        code_keywords = ["code", "script", "example", "snippet", "program", "implementation"]
        return any(word in question.lower() for word in code_keywords)
    
    # Search web if local fails
    def _search_web(self, question):
        web_result = self.web_search.run(question)

        if not web_result:
            return "No relevant information found from local documents or web search."

        if isinstance(web_result, list):
            if all(isinstance(item, dict) for item in web_result):
                return "\n\n".join([item.get("content", "") for item in web_result])
            else:
                return "\n\n".join(str(item) for item in web_result)

        if isinstance(web_result, dict):
            return web_result.get("content", str(web_result))

        return str(web_result)
    
    # Get context from local DB, else web
    def _get_context(self, question):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        local_result = retriever.invoke(question)

        if not local_result:
            print("[INFO] No relevant local docs found. Using web search...")
            return self._search_web(question)

        return "\n\n".join(doc.page_content for doc in local_result)

    # Search function with LLM chain
    def search(self):
        extra_instructions = """
If the answer requires code:
- Provide complete runnable code inside triple backticks ```language
- Include all necessary imports
- Give a short explanation after the code
""" if self._needs_code(self.query) else ""
        chain = (
            RunnableParallel({
                "history": lambda _: self.memory.load_memory_variables({})["history"],
                "context": lambda x: self._get_context(x["question"]),
                "question": RunnablePassthrough(),
                "extra_instructions": lambda x: extra_instructions
            })
            | self.prompt_template
            | self.llm  
            | StrOutputParser()
        )

        return chain.invoke({"question": self.query})
    


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
