from sentence_transformers import SentenceTransformer

# Chargement du modèle SBERT
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Textes à vectoriser
textes = ["Le chat dort sur le tapis.", "Le chien joue dans le jardin."]

# Obtention des embeddings
embeddings = model.encode(textes)

print(embeddings)

import numpy as np

# Vecteurs des deux textes
vec1 = embeddings[0]
vec2 = embeddings[1]

# Calcul de la similarité cosinus
similarité = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print(f"Similarité : {similarité}")