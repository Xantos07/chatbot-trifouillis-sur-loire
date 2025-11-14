import faiss
import numpy as np
import pickle
from embeddings import embed

def main():
    print("Chargement de l'index existant...")
    
    # Charger l'index et les métadonnées
    index = faiss.read_index("faiss_index.idx")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Index chargé : {index.ntotal} vecteurs")
    print(f"Métadonnées chargées : {len(metadata)} entrées\n")
    
    while True:
        print("="*60)
        question = input("\n Posez votre question (ou 'quit' pour quitter) : ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("++")
            break
        
        rechercher_segments_pertinents(question, index, metadata, k=3)


def rechercher_segments_pertinents(question, index, metadata, k=3):
    """
    Recherche les segments les plus pertinents par rapport à la question posée.
    Args:
        question (str): La question posée par l'utilisateur
        k (int): Nombre de segments à récupérer
    Returns:
        list: Liste des segments pertinents
    """
    # Obtenir l'embedding de la question
    question_embedding = embed(question)
    question_embedding = np.array([question_embedding]).astype('float32')

    print("\n Top k segments pertinents pour la question :", question)

    # Recherche dans l'index
    distances, indices = index.search(question_embedding, k)
    # Récupération des segments pertinents

    segments_pertinents = [] # [chunks[i] for i in indices[0]]

    for i, idx in enumerate(indices[0]):
        meta = metadata[idx]
        distance = distances[0][i]

        print(f"{'='*60}")
        print(f"Résultat #{i+1} (distance: {distance:.4f})")
        print(f"{meta['source']}")
        print(f" Chunk : {meta['chunk_id'] + 1}/{meta['total_chunks']}")
        print(f"Texte :")
        print(f" {meta['text'][:100]}...")
        segments_pertinents.append(meta)


    return segments_pertinents

if __name__ == "__main__":
    main()