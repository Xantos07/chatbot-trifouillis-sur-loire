from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

corpus = [
    "Le projet de rénovation de la mairie avance.",
    "Une nouvelle école sera inaugurée le mois prochain.",
    "Les travaux de la route principale sont terminés.",
    "La bibliothèque municipale organise une exposition.",
    "Un plan de développement durable a été adopté.",
    "Quand ouvre la nouvelle école ?",
    "Quels sont les horaires de la bibliothèque ?",
    "Où se trouvent les pistes cyclables ?",
    "Comment participer aux réunions municipales ?",
    "Qui est responsable des espaces verts ?",
    "Quels sont les projets pour le centre-ville ?",
    "Y a-t-il des événements prévus ce week-end ?",
    "quelles sont les fermes de poximité ?",
    "quelles sont les especes d'oiseaux locales ?",
    "comment fonctionne le tri des déchets ?"
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Réduction de dimensionnalité avec t-SNE
tsne = TSNE(n_components=2, perplexity=1, max_iter=1000)  # <-- Changé n_iter en max_iter
embeddings_2d = tsne.fit_transform(embeddings)

# previsualisation des embeddings 
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='blue')
for i, text in enumerate(corpus):
    plt.annotate(text, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9)
plt.title("Visualisation des embeddings avec t-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()