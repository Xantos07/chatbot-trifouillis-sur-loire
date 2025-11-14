from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialisation du splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def split_documents(documents):
    """
    Découpe un dict {nom: texte} en segments avec métadonnées.

    Args:
        documents (dict): {nom_fichier: texte}

    Returns:
        tuple: (liste_textes, liste_métadonnées)
    """
    all_chunks = []
    all_metadata = []
    
    for doc_name, doc_text in documents.items():
        segments = text_splitter.split_text(doc_text)
        
        print(f"Document : {doc_name}")
        print(f"  → {len(segments)} segments créés")
        print("-" * 60)
        
        for i, segment in enumerate(segments):
            all_chunks.append(segment)
            all_metadata.append({
                "source": doc_name,
                "chunk_id": i,
                "total_chunks": len(segments)
            })
    
    return all_chunks, all_metadata