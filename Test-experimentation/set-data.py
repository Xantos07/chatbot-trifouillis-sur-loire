from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
from dotenv import load_dotenv
from pathlib import Path

# Charger le .env 
load_dotenv(dotenv_path=Path('.') / '.env')

# Initialisation du client avec votre clé API 
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise EnvironmentError("MISTRAL_API_KEY introuvable. Ajoute-la dans .env ou dans les variables d'environnement.")

client = MistralClient(api_key=api_key)

def embed_text(text):
    # Appel de l'API d'embedding
    embeddings_batch = client.embeddings(
        model="mistral-embed",
        input=text
    )
    return embeddings_batch.data[0].embedding

import faiss
import numpy as np

# Charger tous les fichiers markdown du dossier outputs
documents = []
output_dir = Path('outputs')

if not output_dir.exists():
    raise FileNotFoundError("Le dossier 'outputs' n'existe pas. Lance d'abord main.py pour convertir les documents.")

import subprocess

for fichier in output_dir.glob('*.md'):
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            contenu = f.read()
        if contenu.strip():
            documents.append(contenu)
            print(f"Chargé : {fichier.name}")
    except Exception as e:
        print(f"Fichier ignoré ({e.__class__.__name__}) : {fichier.name} - {e}")


if not documents:
    raise ValueError("Aucun document markdown trouvé dans 'outputs'. Lance d'abord main.py.")

print(f" {len(documents)} documents chargés depuis outputs/")

# Génération des embeddings pour chaque document
print("Génération des embeddings via Mistral API...")
embeddings = np.array([embed_text(doc) for doc in documents])

# Initialisation de l'index Faiss
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Ajout des embeddings à l'index
index.add(embeddings)

# Sauvegarde de l'index sur le disque
faiss.write_index(index, "faiss_index.idx")

# Sauvegarder aussi les documents pour la recherche
import pickle
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"Index Faiss et documents sauvegardés ({len(documents)} documents indexés)")
print("Fichiers créés : faiss_index.idx + documents.pkl")




