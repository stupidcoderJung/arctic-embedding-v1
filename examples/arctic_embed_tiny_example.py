import torch
import lancedb
from transformers import AutoTokenizer, AutoModel
import numpy as np
import gc
import os


def setup_mps_device():
    """Setup MPS device for Apple Silicon"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    
    return device


def load_arctic_embed_tiny_model(device):
    """Load the Snowflake Arctic Embed Tiny model"""
    model_name = "Snowflake/snowflake-arctic-embed-xs"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return tokenizer, model


def encode_text(tokenizer, model, texts, device, max_length=512):
    """Encode text using the Arctic Embed model"""
    # Tokenize input texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length
    ).to(device)
    
    # Generate embeddings
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**inputs)
        
        # Use mean pooling to get sentence embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Move embeddings to CPU and convert to numpy
    embeddings_cpu = embeddings.cpu().numpy()
    
    # Explicitly delete GPU tensors to free memory
    del inputs, outputs, embeddings
    
    return embeddings_cpu


def setup_lancedb_connection(uri="./.lancedb"):
    """Setup connection to LanceDB"""
    db = lancedb.connect(uri)
    return db


def create_table_with_embeddings(db, table_name, embeddings, texts):
    """Create a table in LanceDB with embeddings and associated text"""
    # Prepare data for insertion
    data = []
    for i, (emb, txt) in enumerate(zip(embeddings, texts)):
        data.append({
            "id": i,
            "vector": emb.tolist(),
            "text": txt
        })
    
    # Create or overwrite the table
    table = db.create_table(table_name, data, mode="overwrite")
    
    return table


def search_similar_vectors(table, query_embedding, limit=5):
    """Search for similar vectors in the table"""
    # Convert query embedding to list if it's a numpy array
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    
    # Perform vector search
    results = table.search(query_embedding).limit(limit).to_list()
    
    return results


def main():
    """Main function demonstrating Arctic Embed Tiny with LanceDB integration"""
    print("Starting Arctic Embed Tiny with MPS and LanceDB example...")
    
    # Setup MPS device
    device = setup_mps_device()
    
    # Load the model
    print("\nLoading Arctic Embed Tiny model...")
    tokenizer, model = load_arctic_embed_tiny_model(device)
    
    # Sample texts to encode
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a versatile programming language",
        "Natural language processing enables computers to understand human language",
        "Vector databases are essential for semantic search applications"
    ]
    
    print(f"\nEncoding {len(sample_texts)} texts...")
    embeddings = encode_text(tokenizer, model, sample_texts, device)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Setup LanceDB connection
    print("\nSetting up LanceDB connection...")
    db = setup_lancedb_connection()
    
    # Create a table with embeddings
    table_name = "arctic_embeddings_demo"
    print(f"Creating table '{table_name}' with embeddings...")
    table = create_table_with_embeddings(db, table_name, embeddings, sample_texts)
    
    # Query example
    query_text = ["Find information about machine learning"]
    print(f"\nEncoding query: {query_text[0]}")
    query_embedding = encode_text(tokenizer, model, query_text, device)
    
    print("Searching for similar vectors...")
    results = search_similar_vectors(table, query_embedding[0], limit=3)
    
    print("\nTop 3 similar results:")
    for i, result in enumerate(results):
        print(f"{i+1}. ID: {result['id']}, Text: {result['text'][:100]}...")
        print(f"   Distance: {result['_distance']:.4f}")
    
    # Clean up resources
    print("\nCleaning up resources...")
    del tokenizer, model
    if device.type == "mps":
        # Clear MPS cache to free memory
        torch.mps.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    print("\nResources cleaned up successfully!")
    

if __name__ == "__main__":
    main()