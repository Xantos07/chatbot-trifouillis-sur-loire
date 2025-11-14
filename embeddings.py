from mistralai.client import MistralClient
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path('.') / '.env')

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise EnvironmentError("MISTRAL_API_KEY introuvable.")

client = MistralClient(api_key=api_key)

def embed_chunks(chunks, batch_size=32):
    """
    Transforme une liste de chunks en vecteurs (par batch pour optimiser).
    
    Args:
        chunks: Liste de textes à vectoriser
        batch_size: Nombre de chunks par requête API (max recommandé: 32)
    
    Returns:
        Liste de vecteurs
    """
    vecteurs = []
    
    # Traiter par batch pour réduire le nombre d'appels API
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        print(f"Vectorisation batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
        
        # Un seul appel API pour tout le batch
        response = client.embeddings(
            model="mistral-embed",
            input=batch  # ← Liste de textes
        )
        
        # Récupération de tous les vecteurs du batch
        for data in response.data:
            vecteurs.append(data.embedding)
    
    return vecteurs

# Utilisation
chunks = ["Le chat dort.", "Python est super.", "Le soleil brille."]
vecteurs = embed_chunks(chunks)

print(f"Nombre de chunks: {len(vecteurs)}")
print(f"Taille d'un vecteur: {len(vecteurs[0])}")