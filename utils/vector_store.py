import faiss
import numpy as np

def create_index(vectors):
    """
    Crée un index FAISS à partir d'une liste de vecteurs.

    Args:
        vectors (list): Liste de vecteurs (listes ou arrays)

    Returns:
        faiss.Index: Index FAISS créé
    """

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    
    np_vectors = np.array(vectors).astype('float32')
    index.add(np_vectors)


    return index