# utils/rag_utils.py
import streamlit as st
import os
import tempfile
import logging
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
)
from .config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_document(uploaded_file):
    """Loads content from an uploaded file object based on its type."""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, uploaded_file.name)

    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        loader = None

        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_path, encoding='utf-8') # Specify encoding
        elif file_extension == ".md":
            # Requires 'unstructured' and potentially 'markdown' library
            try:
                loader = UnstructuredMarkdownLoader(temp_path)
            except ImportError:
                st.error("Processing .md files requires 'unstructured' and 'markdown'. Install with: pip install unstructured markdown")
                return None
            except Exception as e:
                 st.error(f"Error initializing Markdown loader: {e}")
                 return None
        elif file_extension == ".csv":
             # Requires 'pandas'
            try:
                loader = CSVLoader(temp_path, encoding='utf-8') # Specify encoding
            except ImportError:
                st.error("Processing .csv files requires 'pandas'. Install with: pip install pandas")
                return None
            except Exception as e:
                 st.error(f"Error initializing CSV loader: {e}")
                 return None
        elif file_extension in [".xlsx", ".xls"]:
            # Requires 'unstructured', 'openpyxl', and potentially 'tabulate'
            try:
                # Use unstructured for better handling of complex sheets
                loader = UnstructuredExcelLoader(temp_path, mode="elements")
            except ImportError:
                st.error("Processing .xlsx/.xls requires 'unstructured' and 'openpyxl'. Install with: pip install unstructured openpyxl")
                return None
            except Exception as e:
                 st.error(f"Error initializing Excel loader: {e}")
                 return None
        else:
            st.warning(f"Unsupported file type: {file_extension}. Skipping {uploaded_file.name}")
            return None # Return None if unsupported

        if loader:
            logging.info(f"Loading document: {uploaded_file.name} with {type(loader).__name__}")
            docs = loader.load()
            # Add source metadata which is useful for RAG
            for doc in docs:
                 doc.metadata["source"] = uploaded_file.name
            logging.info(f"Successfully loaded {len(docs)} documents from {uploaded_file.name}")

    except Exception as e:
        logging.error(f"Error loading/processing file {uploaded_file.name}: {e}", exc_info=True)
        st.error(f"Failed to load {uploaded_file.name}: {e}")
        docs = None # Ensure docs is None on error
    finally:
        temp_dir.cleanup() # Clean up temporary file

    return docs


def split_documents(documents):
    """Splits loaded documents into chunks."""
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Helps identify chunk position
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def get_openai_embeddings():
    """Initializes OpenAI embeddings, checking for API key."""
    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API Key not found. Please set it in the sidebar.")
        return None
    try:
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        # Perform a dummy embed to check key validity early
        embeddings.embed_query("test")
        return embeddings
    except Exception as e:
        st.error(f"Failed to initialize OpenAI Embeddings. Check API Key. Error: {e}")
        logging.error(f"OpenAI Embeddings initialization failed: {e}")
        return None


def create_vector_store(chunks):
    """Creates a FAISS vector store from document chunks."""
    if not chunks:
        st.warning("No document chunks to index.")
        return None

    embeddings = get_openai_embeddings()
    if not embeddings:
        return None # Error handled in get_openai_embeddings

    try:
        with st.spinner("Creating vector store... This may take a moment."):
            vector_store = FAISS.from_documents(chunks, embeddings)
        logging.info("FAISS vector store created successfully.")
        # Optional: Save the index locally
        # if not os.path.exists(VECTOR_STORE_DIR):
        #     os.makedirs(VECTOR_STORE_DIR)
        # vector_store.save_local(VECTOR_STORE_DIR)
        # logging.info(f"Vector store saved to {VECTOR_STORE_DIR}")
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        logging.error(f"FAISS creation failed: {e}", exc_info=True)
        return None

# Optional: Function to load from disk if you implement saving
# def load_vector_store():
#     embeddings = get_openai_embeddings()
#     if not embeddings: return None
#     if os.path.exists(VECTOR_STORE_DIR):
#         try:
#             vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True) # Be cautious with deserialization
#             logging.info("Loaded vector store from disk.")
#             return vector_store
#         except Exception as e:
#             st.error(f"Failed to load vector store from {VECTOR_STORE_DIR}: {e}")
#             logging.error(f"Failed loading FAISS index: {e}")
#             return None
#     else:
#         logging.info("No existing vector store found on disk.")
#         return None

def get_retriever(vector_store, k=4):
    """Creates a retriever from the vector store."""
    if not vector_store:
        return None
    # k determines how many relevant chunks to retrieve
    return vector_store.as_retriever(search_kwargs={"k": k})

def format_retrieved_docs(docs):
    """Formats retrieved documents for inclusion in the prompt."""
    if not docs:
        return "No relevant documents found in the local index."
    # Simple formatting, including source
    formatted = ["--- Retrieved Context From Your Documents ---"]
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', None) # Specific to PDFs often
        location = f" (Source: {source}{f', Page: {page + 1}' if page is not None else ''})" # Add 1 to page for display
        formatted.append(f"Chunk {i+1}{location}:\n{doc.page_content}")
        formatted.append("---") # Separator between chunks
    formatted.append("--- End Retrieved Context ---")
    return "\n".join(formatted)