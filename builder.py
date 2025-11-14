from utils.data_loader import load_documents_from_dir
from indexer import split_documents
from embeddings import embed_chunks


def main():
    print("Chargement des documents depuis le répertoire 'inputs/'...")
    documents = load_documents_from_dir("inputs")
    print(f"{len(documents)} documents chargés.")

    print("Découpage des documents en segments...")
    segments = split_documents(documents)
    print(f"{len(segments)} segments créés.")

    print("Embedding des segments...")
    embeddings = embed_chunks(segments)
    print(f"{len(embeddings)} embeddings générés.")

if __name__ == "__main__":
    main()