import streamlit as st
import os 
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

import logging
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join('.', '.env'))

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Chatbot Trifouillis-sur-Loire",
    page_icon="🤖")

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("MISTRAL_API_KEY introuvable. Ajoute-la dans .env ou dans les variables d'environnement.")
    st.stop()

try: 
    client = MistralClient(api_key=api_key)
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral: {e}")
    st.stop()   

# Charger le prompt système depuis garde-fou.py
def load_system_prompt():
    try:
        with open('garde-fou.py', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """### RÔLE :
Vous êtes l'assistant virtuel officiel de la mairie de Trifouillis-sur-Loire. Agissez comme un agent d'accueil numérique compétent et bienveillant.

### COMPORTEMENT & STYLE :
Ton : Formel, courtois, patient, langage simple et accessible.
Précision : Informations exactes et vérifiées.
Ambiguïté : Demander poliment des précisions si la question est vague.

### INTERDICTIONS STRICTES :
Ne JAMAIS inventer d'informations.
Ne JAMAIS fournir d'information non vérifiée.
Ne JAMAIS donner d'avis personnel ou politique."""

if 'messages' not in st.session_state:
    system_prompt = load_system_prompt()
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

def generate_prompt_session(messages, max_messages=10):
    """
        Construire le prompt pour l'api mistral en utilisant les messages récents.

        Args:
        messages (list): Liste des messages de la session.
        max_messages (int): Nombre maximum de messages à inclure dans le prompt.

        Returns: 
        list[ChatMessage]: Liste des messages formatés pour l'API Mistral.
    """

    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

    format_messages = [
        ChatMessage(role=msg["role"], content=msg["content"]) for msg in recent_messages
    ]

    logging.info(f"Prompt construit avec {len(format_messages)} messages.")
    return format_messages

def generate_response(messages):
    """
        Générer une réponse en utilisant l'API Mistral.

        Args:
        messages (list): Liste des messages de la session.

        Returns:
        str: Réponse générée par le modèle.
    """
    try:
        prompt = generate_prompt_session(messages)

        # Utiliser la bonne méthode de l'API Mistral
        response = client.chat(
            model="mistral-small-latest",  # Modèle correct
            messages=prompt,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
        )

        answer = response.choices[0].message.content
        logging.info("Réponse générée avec succès.")
        return answer

    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse: {e}")
        # Afficher l'erreur complète pour debug
        st.error(f"Erreur détaillée : {str(e)}")
        return f"Désolé, une erreur est survenue : {str(e)}"
    
st.title("🤖 Chatbot de la Mairie de Trifouillis-sur-Loire")
st.caption("Posez vos questions sur les services municipaux.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Écrivez votre message ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    prompt_messages_for_api = generate_prompt_session(st.session_state.messages)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...")  

        reponse_content = generate_response(st.session_state.messages)
        message_placeholder.text(reponse_content)

    st.session_state.messages.append({"role": "assistant", "content": reponse_content})

if st.button("Réinitialiser la conversation"):
    system_prompt = load_system_prompt()
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.rerun()