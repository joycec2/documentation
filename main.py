import chromadb
from dotenv import load_dotenv
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import ListIndex, SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleNodeParser
import argparse

def configure_llm(model_name: str, temperature: float = 0.2, max_token: int = 2048, request_timeout: float = 360.0):
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
        transformations=[SimpleNodeParser(chunk_size=1024, chunk_overlap=20)]
    )

    return index

def run_query_with_parameters(
    model_name: str,
    embed_model_name: str,
    query: str,
    data_path: str,
    temperature: float = 0.2,
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
    configure_llm(model_name, temperature=temperature)

    # Initialize Chroma collection
    chroma_collection = initialize_chroma_collection(collection_name, ephemeral)

    # Load documents
    documents = load_documents(data_path)

    # Create vector store
    vector_store = create_vector_store(chroma_collection)

    # Build index
    index = build_index(documents, vector_store, embed_model_name)

    # Run query
    query_engine = index.as_query_engine(similarity_top_k=15)
    response = query_engine.query(query)

    return response

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a query on a document index.")
    parser.add_argument("--model_name", type=str, default="llama3.1:8b-instruct-fp16", help="Name of the LLM model.")
    parser.add_argument("--embed_model_name", type=str, default="snowflake-arctic-embed", help="Name of the embedding model.")
    parser.add_argument("--query", type=str, required=True, help="Query prompt to run against the index.")
    parser.add_argument("--data_path", type=str, default="./data/ravens_web_official_news_10_7_10_14", help="Path to the data directory.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for the LLM.")
    parser.add_argument("--collection_name", type=str, default="ravens_oct_news", help="Name of the Chroma collection.")
    parser.add_argument("--ephemeral", type=bool, default=True, help="Whether to use an ephemeral Chroma client.")

    args = parser.parse_args()

    # Run query with parameters
    response = run_query_with_parameters(
        model_name=args.model_name,
        embed_model_name=args.embed_model_name,
        query=args.query,
        data_path=args.data_path,
        temperature=args.temperature,
        collection_name=args.collection_name,
        ephemeral=args.ephemeral
    )

    # Print the response
    print(response)

    # python main.py --query "What's going on with Baltimore Ravens in October?"
