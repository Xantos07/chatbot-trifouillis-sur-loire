import faiss
import numpy as np
import pickle
import os
from embeddings import embed
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

# Variables globales pour l'index
index = None
metadata = None

def main():
    global index, metadata
    
    print("Chargement de l'index existant : ")
    
    # Charger l'index et les métadonnées
    index = faiss.read_index("faiss_index.idx")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Index chargé : {index.ntotal} vecteurs")
    print(f"Métadonnées chargées : {len(metadata)} entrées\n")
    
    # Historique de conversation
    messages = []
    
    # Boucle interactive
    while True:
        print("="*60)
        question = input("\n🤖 Posez votre question (ou 'quit' pour quitter) : ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Au revoir !")
            break
        
        # Ajouter la question à l'historique
        messages.append({"role": "user", "content": question})
        
        # Construire le prompt avec contexte
        formatted_messages = construire_prompt_session(messages, question, max_messages=5)
        
        # Appeler l'API Mistral pour générer la réponse
        print("\n Génération de la réponse...")
        try:
            response = client.chat(
                model="mistral-small-latest",
                messages=formatted_messages
            )
            
            reponse = response.choices[0].message.content
            
            # Ajouter la réponse à l'historique
            messages.append({"role": "assistant", "content": reponse})
            
            # Afficher la réponse
            print(f"\n Réponse :\n{reponse}\n")
            
        except Exception as e:
            print(f" Erreur lors de la génération : {e}")
            # Retirer la question de l'historique en cas d'erreur
            messages.pop()


def rechercher_segments_pertinents(question, k=3):
    """
    Recherche les segments les plus pertinents par rapport à la question posée.
    
    Args:
        question (str): La question posée par l'utilisateur
        k (int): Nombre de segments à récupérer
    
    Returns:
        list: Liste des textes des segments pertinents
    """
    global index, metadata
    
    # Obtenir l'embedding de la question
    question_embedding = embed(question)
    question_embedding = np.array([question_embedding]).astype('float32')

    # Recherche dans l'index
    distances, indices = index.search(question_embedding, k)
    
    # Afficher les résultats (optionnel, pour debug)
    print(f"\n Top {k} segments pertinents trouvés :")
    
    segments_pertinents = []
    for i, idx in enumerate(indices[0]):
        meta = metadata[idx]
        distance = distances[0][i]

        print(f"  {i+1}. {meta['source']} (chunk {meta['chunk_id']+1}/{meta['total_chunks']}) - distance: {distance:.4f}")
        
        # Retourner uniquement le TEXTE pour le prompt
        segments_pertinents.append(meta['text'])

    print()  # Ligne vide
    return segments_pertinents


def construire_prompt_session(messages, question=None, max_messages=5):
    """
    Construit un prompt enrichi avec les segments pertinents et l'historique récent.
    
    Args:
        messages (list): Historique des messages
        question (str): Question actuelle (pour rechercher le contexte)
        max_messages (int): Nombre max de messages récents à inclure
    
    Returns:
        list: Liste de ChatMessage formatés pour l'API Mistral
    """
    # Limiter le nombre de messages récents
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    # Si une question est fournie, rechercher les segments pertinents
    context_segments = []
    if question:
        context_segments = rechercher_segments_pertinents(question, k=3)
    
    # Création du système prompt avec le contexte
    system_prompt = "Vous êtes l'assistant virtuel de la mairie de Trifouillis-sur-Loire. "
    
    if context_segments:
        system_prompt += "Veuillez utiliser les informations suivantes pour répondre à la question:\n\n"
        system_prompt += "CONTEXTE:\n"
        for i, segment in enumerate(context_segments):
            system_prompt += f"[Document {i+1}]\n{segment}\n\n"
        system_prompt += "\nRÈGLES:\n"
        system_prompt += "- Répondez en vous basant UNIQUEMENT sur les informations fournies ci-dessus.\n"
        system_prompt += "- Si les informations ne permettent pas de répondre précisément, indiquez-le clairement.\n"
        system_prompt += "- Soyez concis et précis.\n"
        system_prompt += "- Citez la source si pertinent (ex: 'Selon le règlement municipal...').\n"
    
    # Création des messages formatés
    formatted_messages = [ChatMessage(role="system", content=system_prompt)]
    
    # Ajout des messages récents (sans le dernier qui est déjà la question actuelle)
    for msg in recent_messages[:-1]:  # Exclure le dernier (question actuelle)
        formatted_messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
    
    # Ajouter la question actuelle
    if question:
        formatted_messages.append(ChatMessage(role="user", content=question))
    
    return formatted_messages


if __name__ == "__main__":
    main()