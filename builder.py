import os
import faiss
from utils.data_loader import load_documents_from_dir
from indexer import split_documents
from embeddings import embed_chunks
from utils.vector_store import create_index
import pickle

def main():
    print("Chargement des documents depuis le répertoire 'inputs/'...")
    documents = load_documents_from_dir("inputs")
    print(f"{len(documents)} documents chargés.")

    print("Découpage des documents en chunks...")
    chunks, metadata = split_documents(documents)
    print(f"{len(chunks)} chunks créés.")

    print("Embedding des chunks...")
    embeddings, metadata_enrichie = embed_chunks(chunks, metadata)
    print(f"{len(embeddings)} embeddings générés.")

    print("\n Création de l'index Faiss...")
    index = create_index(embeddings) 
    print(f"Faiss {index.ntotal} vecteurs ajoutés à l'index")

    # Temporaire
    faiss.write_index(index, "faiss_index.idx")
    print("Index sauvegardé dans 'faiss_index.idx'")
    with open("metadata.pkl", "wb") as f:
     pickle.dump(metadata_enrichie, f)

    print("sauvegardées dans 'metadata.pkl'")

    # petit test juste pour check
    print("\n Aperçu des données indexées :")
    for i in range(min(3, len(embeddings))):
        print(f"  Vecteur {i} → {metadata_enrichie[i]['source']} (chunk {metadata_enrichie[i]['chunk_id']})")
        print(f"    Texte : {metadata_enrichie[i]['text'][:80]}...")

if __name__ == "__main__":
    main()