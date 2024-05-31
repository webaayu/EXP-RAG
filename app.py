import streamlit as st
import zipfile
import os
import tempfile
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.text_splitter import PlainTextExtractor
from chromadb.config import Settings

# Connect to Chroma DB with SQLite
settings = Settings(
    chroma_db_impl="sqlite",
    sqlite_db_path="chroma_db.sqlite"
)
chroma_client = Chroma(client_settings=settings)
collection_name = "text_collection"

# Create or get the Chroma collection
if collection_name not in chroma_client.list_collections():
    collection = chroma_client.create_collection(collection_name)
else:
    collection = chroma_client.get_collection(collection_name)

# Load models
llama_model = Ollama(model_name="llama3")
embedding_model = HuggingFaceEmbeddings()

# Define the Streamlit app
def main():
    st.title("Zip File Upload and Content Generation")

    uploaded_file = st.file_uploader("Upload a zip file", type="zip")
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
                extracted_files = zip_ref.namelist()

            st.write("Extracted files:")
            for file in extracted_files:
                st.write(file)

            all_texts = []
            text_extractor = PlainTextExtractor()
            for file in extracted_files:
                file_path = os.path.join(tmp_dir, file)
                extracted_text = text_extractor.extract(file_path)
                if extracted_text:
                    all_texts.append(extracted_text)

            st.write("Processing texts...")
            text_splitter = RecursiveCharacterTextSplitterr()
            texts = text_splitter.split_texts(all_texts)

            st.write("Generating embeddings and storing in Chroma DB...")
            embeddings = embedding_model.encode(texts)
            for i, embedding in enumerate(embeddings):
                collection.add(embedding=embedding, document_id=str(i), document_metadata={"text": texts[i]})

            query = st.text_input("Enter your query:")
            if query:
                query_embedding = embedding_model.encode([query])[0]
                results = collection.search(embedding=query_embedding, top_k=5)
                relevant_texts = [result['document_metadata']['text'] for result in results]

                st.write("Generating content based on relevant texts...")
                response = llama_model(relevant_texts)
                st.write(response)

if __name__ == "__main__":
    main()
