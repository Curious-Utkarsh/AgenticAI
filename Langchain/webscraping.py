import os
from dotenv import load_dotenv
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["USER_AGENT"] = "MyRAGApp/1.0"

# -----------------------------
# Custom Embedding Wrapper
# -----------------------------
class SafeGeminiEmbeddings(Embeddings):
    def __init__(self, model: str):
        self._model = GoogleGenerativeAIEmbeddings(model=model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Embed one at a time to avoid batching bug in preview model
        return [self._model.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)

# -----------------------------
# 1. Load data from website
# -----------------------------
url = "https://www.mechamindlabs.com"
loader = WebBaseLoader(url)
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# -----------------------------
# 2. Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks")

# -----------------------------
# 3. Create embeddings (fixed)
# -----------------------------
embeddings = SafeGeminiEmbeddings(model="gemini-embedding-2-preview")

# -----------------------------
# 4. Sanity check
# -----------------------------
texts = [doc.page_content for doc in docs]
vectors = embeddings.embed_documents(texts)
print("Total vectors:", len(vectors))        # must match len(docs)
print("Vector dimension:", len(vectors[0]))  # 3072 for this model
print("First vector sample:", vectors[0][:5])

# -----------------------------
# 5. Create vector DB + query
# -----------------------------
db = Chroma.from_documents(docs, embeddings)

results = db.similarity_search("What does MechaMind Labs do?")
for r in results:
    print(r.page_content)
    print("---")