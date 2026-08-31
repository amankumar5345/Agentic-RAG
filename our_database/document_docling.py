import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from docling.document_converter import DocumentConverter

def deep_clean_text(text):
    return re.sub(r'[\x00-\x1F\x7F]', '', text)


def build_permanent_vector_db(data_folder):
    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        return None

    document_list = []
    for filename in [f for f in os.listdir(data_folder) if f.endswith(".pdf")]:
        document_list.append(os.path.join(data_folder, filename))
    
    converter = DocumentConverter()
    for file_path in document_list:
        print(f"Processing {file_path}...")
    
        # Convert the current PDF
        result = converter.convert(file_path)
        markdown_output = result.document.export_to_markdown()
        
        # Use 'a' mode to APPEND to the file instead of overwriting
        with open("all_contents.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- Source: {file_path} ---\n") # Added for clarity
            f.write(markdown_output)
            f.write("\n\n")
        



if __name__ == "__main__":
    data_source = "our_database\database_documents"
    database_save = "our_database\chroma_database"

    os.makedirs(data_source, exist_ok=True)
    build_permanent_vector_db(database_save)