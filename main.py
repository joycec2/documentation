import chromadb
from dotenv import load_dotenv
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import ListIndex, SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleNodeParser

def configure_llm(model_name: str, temperature: float = 0.2, max_token: int = 1000000, request_timeout: float = 360.0):
    """Configure the LLM settings."""
    Settings.llm = Ollama(
        model=model_name,
        request_timeout=request_timeout,
        temperature=temperature,
        max_token=max_token
    )

def initialize_chroma_collection(collection_name: str, ephemeral: bool = True):
    """
    Initialize a Chroma collection.
    If 'ephemeral' is True, uses an EphemeralClient; otherwise, you could use PersistentClient.
    """
    if ephemeral:
        chroma_client = chromadb.EphemeralClient()
    else:
        chroma_client = chromadb.Client()

    existing_collections = chroma_client.list_collections()

    if collection_name in [collection.name for collection in existing_collections]:
        chroma_collection = chroma_client.get_collection(collection_name)
        print(f"Using existing collection '{collection_name}'.")
    else:
        chroma_collection = chroma_client.create_collection(collection_name)
        print(f"Created new collection '{collection_name}'.")

    return chroma_collection

def load_documents(directory_path: str):
    """Load documents from a specified directory."""
    return SimpleDirectoryReader(directory_path).load_data()

def create_vector_store(chroma_collection):
    """Create a ChromaVectorStore from a given Chroma collection."""
    return ChromaVectorStore(chroma_collection=chroma_collection)

def build_index(documents, vector_store, embed_model_name: str):
    """
    Build a VectorStoreIndex from the provided documents and vector store.
    Uses the specified embedding model.
    """
    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        ollama_additional_kwargs={"prostatic": 0},
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[SimpleNodeParser(chunk_size=512, chunk_overlap=20)]
    )

    return index

def run_query_with_parameters(
    model_name: str,
    embed_model_name: str,
    query: str,
    data_path: str,
    collection_name: str = "default_collection",
    ephemeral: bool = True
):
    """
    Run a query with specified parameters, including model name, embedding model, prompt, and data path.
    Returns the response.
    """
    # Load environment variables
    load_dotenv()

    # Configure LLM
    configure_llm(model_name)

    # Initialize Chroma collection
    chroma_collection = initialize_chroma_collection(collection_name, ephemeral)

    # Load documents
    documents = load_documents(data_path)

    # Create vector store
    vector_store = create_vector_store(chroma_collection)

    # Build index
    index = build_index(documents, vector_store, embed_model_name)

    # Run query
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    return response

if __name__ == "__main__":
    # Define parameters
    model_name = "llama3.1:8b-instruct-fp16"
    embed_model_name = "snowflake-arctic-embed"
    query = "What's new with Baltimore Ravens special teams?"
    data_path = "./data/ravens_web_official_news_10_7_10_14"

    # Run query with parameters
    response = run_query_with_parameters(
        model_name=model_name,
        embed_model_name=embed_model_name,
        query=query,
        data_path=data_path,
        collection_name="ravens",
        ephemeral=True
    )

    # Print the response
    print(response)
