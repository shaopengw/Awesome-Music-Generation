import numpy as np
import faiss
import time
import os

def build_hnsw_index(data, M=32, efConstruction=128):
    """
    Build HNSW index.

    Args:
        data (np.ndarray): Dataset with shape (n_samples, d).
        M (int): Number of connections in HNSW.
        efConstruction (int): ef parameter during construction.

    Returns:
        faiss.IndexHNSWFlat: Built HNSW index.
    """
    d = data.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction

    start_time = time.time()
    index.add(data)
    end_time = time.time()

    print(f"HNSW index construction time: {end_time - start_time:.2f} seconds")
    return index

def evaluate_index(audio_index, melody_index, audio_queries, k=1):
    """
    Evaluate index performance.

    Args:
        audio_index (faiss.Index): Audio index.
        melody_index (faiss.Index): Melody index.
        audio_queries (np.ndarray): Query vectors with shape (n_queries, d).
        k (int): Number of nearest neighbors.

    Returns:
        dict: Evaluation results with search times.
    """
    # Evaluate audio to audio query
    audio_to_audio_start_time = time.time()
    _, _ = audio_index.search(audio_queries, k)
    audio_to_audio_search_time = time.time() - audio_to_audio_start_time

    # Evaluate audio to melody query
    audio_to_melody_start_time = time.time()
    _, _ = melody_index.search(audio_queries, k)
    audio_to_melody_search_time = time.time() - audio_to_melody_start_time

    return {
        'k': k,
        'audio_to_audio_search_time': audio_to_audio_search_time,
        'audio_to_melody_search_time': audio_to_melody_search_time,
    }

def save_index(index, index_path):
    """
    Save FAISS index to specified path.

    Args:
        index (faiss.Index): Index to save.
        index_path (str): File path to save the index.
    """
    faiss.write_index(index, index_path)
    print(f"Index saved to: {index_path}")

if __name__ == "__main__":
    # Load data
    melody_data = np.load('Awesome-Music-Generation/MMGen_train/modules/clmp/faiss_indexing/clmp_embeddings/melody.npy')
    print("melody_data loaded")
    audio_data = np.load('Awesome-Music-Generation/MMGen_train/modules/clmp/faiss_indexing/clmp_embeddings/audio.npy')
    print("audio_data loaded")
    audio_queries = np.load('Awesome-Music-Generation/MMGen_train/modules/clmp/faiss_indexing/clmp_embeddings/audio.npy')
    print("audio_queries loaded")
    
    # HNSW parameters
    M = 32
    efConstruction = 80
    k = 1

    print("Building melody HNSW index...")
    melody_index = build_hnsw_index(melody_data, M=M, efConstruction=efConstruction)
    
    print("Building audio HNSW index...")
    audio_index = build_hnsw_index(audio_data, M=M, efConstruction=efConstruction)

    index_info = f"HNSW, M: {M}, efConstruction: {efConstruction}"
    file_suffix = "hnsw"

    # Run validation
    print(f"Running validation with k={k}...")
    validation_result = evaluate_index(audio_index, melody_index, audio_queries, k=k)

    # Output validation results
    print(f"\nIndex type: {index_info}, k: {k}")
    print(f"Audio to audio search time: {validation_result['audio_to_audio_search_time']:.6f} seconds")
    print(f"Audio to melody search time: {validation_result['audio_to_melody_search_time']:.6f} seconds")

    # Save index
    save_path = 'Awesome-Music-Generation/MMGen_train/modules/clmp/faiss_indexing/faiss_indices'

    melody_index_path = os.path.join(save_path, f'audio_2_melody_{file_suffix}.faiss')
    print("Saving melody HNSW index...")
    save_index(melody_index, melody_index_path)
