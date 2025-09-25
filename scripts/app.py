import os
import sys
import pprint
import weaviate
import atexit
from weaviate.classes.init import Auth
from flask import Flask, request, jsonify
from flask_cors import CORS

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from dotenv import load_dotenv

# --- 1. ONE-TIME SERVER INITIALIZATION ---

load_dotenv()
print("🚀 Initializing Vajra Bot Server...")

# --- [FIXED] Add checks for essential environment variables ---
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not weaviate_url or not weaviate_api_key or not google_api_key:
    print("🔥🔥🔥 FATAL: Missing one or more environment variables. Check your .env file.")
    sys.exit(1)

# Configure LlamaIndex Settings
Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash-latest")
Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")
print(f"Embedding model set to: {Settings.embed_model.model_name}")
print(f"LLM set to: {Settings.llm.model_config}")

# Global variables
query_engine = None
client = None # Hold the client in a global variable

try:
    index_name = "Curaj_test"
    
    # --- [FIXED] Create the client WITHOUT a 'with' statement ---
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    # --- [FIXED] Register a function to close the client on exit ---
    atexit.register(client.close)
    print("Weaviate client connected. It will be closed on server shutdown.")
    
    data_exists = client.collections.exists(index_name)
    index = None

    print(f"Connecting to existing Weaviate index: '{index_name}'")
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    query_engine = index.as_query_engine(similarity_top_k=5)
    print("✅ Query engine is ready to serve requests.")

except Exception as e:
    print(f"🔥🔥🔥 FATAL: FAILED TO INITIALIZE SERVER! 🔥🔥🔥")
    print(f"Error: {e}")
    if client:
        client.close() # Attempt to close client on failure

# --- 2. FLASK API DEFINITION ---

app = Flask(__name__)
CORS(app) 

@app.route("/api/query", methods=["POST"])
def handle_query():
    """
    API endpoint to handle user queries.
    Expects a JSON body with a "query" key.
    e.g., {"query": "Tell me about the university's vision"}
    """
    global query_engine
    if query_engine is None:
        return jsonify({"error": "Query engine is not initialized. Check server logs for errors."}), 500

    if not request.is_json:
        return jsonify({"error": "Invalid request: body must be JSON"}), 400

    data = request.get_json()
    user_query = data.get("prompt")

    if not user_query or not user_query.strip():
        return jsonify({"error": "Invalid request: 'query' field is missing or empty"}), 400
    
    print(f"Received query: '{user_query}'")

    # Use the globally initialized query engine
    response = query_engine.query(user_query)

    # Format a clean response for your frontend
    formatted_response = {
        "response": response.response.strip(),
        "sources": [
            {
                "file_name": node.metadata.get("file_name", "N/A"),
                "score": f"{node.score:.4f}"
            } for node in response.source_nodes
        ]
    }
    
    return jsonify(formatted_response)

if __name__ == "__main__":
    print("🌍 Starting Flask server on http://localhost:5000")
    # Use host='0.0.0.0' to make it accessible from other devices on your local network
    app.run(host="0.0.0.0", port=5000, debug=False)
