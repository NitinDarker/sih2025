import os
import pprint
import weaviate
from weaviate.classes.config import Configure

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

from llama_index.readers.file import SimpleDirectoryReader
from llama_index.core import Document, VectorStoreIndex, HierarchicalNodeParser
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# ----------------------------
# Initialize LLM and embedding
# ----------------------------
Settings.llm = Ollama(model="llama3.2", request_timeout=300)
embedding_model_name = "nomic-embed-text"
Settings.embed_model = OllamaEmbedding(model_name=embedding_model_name)

print(f"LLM model object: {Settings.llm}")
print(f"Embedding model: {embedding_model_name}")

# ----------------------------
# Collection name in Weaviate
# ----------------------------
index_name = "Curaj"

def run_llama():
    # ----------------------------
    # Connect to local Weaviate
    # ----------------------------
    client = weaviate.connect_to_local(port=8080)
    if not client.is_ready():
        raise RuntimeError("Weaviate is not ready. Make sure Docker container is running.")

    # ----------------------------
    # Create collection if missing
    # ----------------------------
    if not client.collections.exists(index_name):
        print("Creating new collection in Weaviate...")

        vectorizer_config = Configure.Vectorizer.text2vec_ollama(
            api_endpoint="http://host.docker.internal:11434",
            model=embedding_model_name,
        ).dict()

        generative_config = Configure.Generative.ollama(
            api_endpoint="http://host.docker.internal:11434",
            model="llama3.2",
        ).dict()

        client.collections.create(
            name=index_name,
            vector_config=[vectorizer_config],
            generative_config=generative_config,
        )

    # ----------------------------
    # Locate data folders
    # ----------------------------
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dirs = [
        os.path.join(project_root, "sorted_data", "repaired"),
        os.path.join(project_root, "sorted_data", "ocr_texts")
    ]

    # Ensure folders exist
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # ----------------------------
    # Load documents from all folders
    # ----------------------------
    docs = []
    for dir_path in data_dirs:
        reader = SimpleDirectoryReader(input_dir=dir_path, recursive=True)
        docs.extend(reader.load_data())

    if not docs:
        print("[WARN] No documents found in repaired or OCR folders. Please run 'run_extract()' first.")
        return

    # ----------------------------
    # Chunk documents
    # ----------------------------
    splitter = HierarchicalNodeParser.from_defaults()
    for doc in docs:
        doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"
        doc.excluded_embed_metadata_keys = [k for k in doc.metadata if k != "file_name"]
        doc.excluded_llm_metadata_keys = [k for k in doc.metadata if k != "file_name"]

    nodes = splitter.get_nodes_from_documents(docs)
    print(f"✅ Chunked {len(nodes)} nodes.")

    # ----------------------------
    # Store in Weaviate
    # ----------------------------
    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name=index_name
    )
    index = VectorStoreIndex(nodes=nodes, vector_store=vector_store)
    print("✅ Data indexed successfully.")

    # ----------------------------
    # Interactive query engine
    # ----------------------------
    query_engine = index.as_query_engine(similarity_top_k=5)
    print("\n--- Query Engine Ready ---")
    print("Type 'exit' to quit.")

    while True:
        query_text = input("\nAsk a question: ")
        if query_text.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not query_text.strip():
            print("Query cannot be empty.")
            continue

        response = query_engine.query(query_text)
        print("\n--- Response ---")
        pprint.pprint(response.response)

        print("\n--- Source Nodes ---")
        if response.source_nodes:
            for node in response.source_nodes:
                print(f"📄 {node.metadata.get('file_name', 'N/A')} | Score: {node.score:.4f}")
        else:
            print("No source nodes found.")

    client.close()
