# Import dependencies
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

# Load the HTML content
with open("data/BILLS-119hr1enr.html", "r", encoding="utf-8") as file:
    html_content = file.read()

# Parse the HTML content 
soup = BeautifulSoup(html_content, "lxml")  # or "html.parser"
text = soup.get_text(separator=" ", strip=True)
# print(text[:500])  # Preview the extracted text

# Define the embeddings model   
embeddings_model = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL")  # Use your .env value
)

# Set the persistent directory for the embeddings
persist_directory = ".chroma_db"

# Define the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# split the text into chunks
chunks = splitter.split_text(text)
print(f"Number of chunks: {len(chunks)}")
print(chunks[0])  # Preview the first chunk

# Create metadata for each chunk (optional, but recommended for search)
metadatas = [{"chunk_id": i} for i in range(len(chunks))]

# Save embeddings and chunks to ChromaDB
print(f"Using embeddings model: {embeddings_model.model}")
try:
    vector_store = Chroma.from_texts(
        texts=chunks,
        metadatas=metadatas,
        embedding=embeddings_model,
        collection_name="congressional_bill_hr1",
        persist_directory=persist_directory
    )
except Exception as e:
    print("Error during ChromaDB creation:", e)

print(f"Saved {len(chunks)} chunks to ChromaDB at {persist_directory}")
