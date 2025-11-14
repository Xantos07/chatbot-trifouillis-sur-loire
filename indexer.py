from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialisation du splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Taille de chaque segment
    chunk_overlap=200 # Chevauchement entre les segments
)
    
# Découpage du texte
segments = text_splitter.split_text("Votre texte ici")

def split_documents(documents):
    """
    Découpe une liste de documents en segments plus petits.

    Args:
        documents (list): Liste de chaînes de caractères représentant les documents.

    Returns:
        list: Liste des segments découpés.
    """
    all_segments = []
    for doc_name, doc in documents.items():
        segments = text_splitter.split_text(doc)
        all_segments.extend(segments)

        print(f"Document : {doc_name if doc_name else 'Inconnu'}")
        print(f"Document découpé en {len(segments)} segments.")
        print("-" * 40)
    return all_segments