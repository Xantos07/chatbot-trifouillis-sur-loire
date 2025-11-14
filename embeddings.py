from mistralai.client import MistralClient
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path('.') / '.env')

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise EnvironmentError("MISTRAL_API_KEY introuvable.")

client = MistralClient(api_key=api_key)

def embed_chunks(chunks, metadata, batch_size=32):
    """
    Transforme une liste de chunks en vecteurs (par batch pour optimiser).
    
    Args:
        chunks: Liste de textes à vectoriser
        metadata: Liste de métadonnées associées à chaque chunk
        batch_size: Nombre de chunks par requête API (max 32)
    
    Returns:
        vecteurs, metadata
    """
    if len(chunks) != len(metadata):
        raise ValueError("chunks et metadata doivent avoir la même longueur")
    
    vecteurs = []
    metadata_enrichie = []
    
    # Traiter par batch pour réduire le nombre d'appels API
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_metadata = metadata[i:i+batch_size]
        
        print(f"Vectorisation batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
        
        # Un seul appel API pour tout le batch
        response = client.embeddings(
            model="mistral-embed",
            input=batch  # Liste de textes
        )
        
        # Récupération des vecteurs + association avec métadonnées
        for j, data in enumerate(response.data):
            vecteurs.append(data.embedding)
        
            enriched_meta = batch_metadata[j].copy()
            enriched_meta["text"] = batch[j]
            metadata_enrichie.append(enriched_meta)
    
    return vecteurs, metadata_enrichie