import os
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def deep_clean_text(text):
    # Removes non-printable control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return " ".join(cleaned.split())

def build_permanent_vector_db(data_path, database_folder):
    print(f"Loading {data_path}...")
    loader = TextLoader(data_path, encoding='utf-8')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, 
        separators=["\n\n", "\n", " "]
    )
    
    chunks = text_splitter.split_documents(docs)
    
    valid_chunks = []
    for chunk in chunks:
        cleaned_content = deep_clean_text(chunk.page_content)
        if len(cleaned_content) > 10 and re.search(r'[a-zA-Z0-9]', cleaned_content):
            chunk.page_content = cleaned_content
            valid_chunks.append(chunk)

    print(f"Total valid chunks ready for embedding: {len(valid_chunks)}")

    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    
    vector_db = Chroma(
        persist_directory=database_folder,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    batch_size = 20  
    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i : i + batch_size]
        try:
            vector_db.add_documents(batch)
            print(f"Successfully embedded batch {i} to {i + len(batch)}")
        except Exception as e:
            print(f"\n[ERROR] Failed to embed batch {i} to {i + len(batch)}.")
            print(f"Error Message: {e}")
            print(f"Problematic chunk preview: {batch[0].page_content[:200]}\n")
            # Skips the bad batch and continues building the DB
            continue
            
    print("Database build process completed.")
    return vector_db

if __name__ == "__main__":
    DATA_SOURCE = "our_database/all_contents.txt" 
    DATABASE_SAVE_FOLDER = "our_database/chroma_storage"
    os.makedirs(DATABASE_SAVE_FOLDER, exist_ok=True)
    build_permanent_vector_db(DATA_SOURCE, DATABASE_SAVE_FOLDER)