# Étape 1 : Chargement des données (Loading)
from pathlib import Path
from typing import Dict
import PyPDF2 
from docx import Document
import pandas as pd

# Fonction pour charger un type de documents
def load_pdf(path: Path) -> str:
    """ Lit un fichier PDF et retourne son contenu en texte. """
    try: 
        parts = []
        with open(path,"rb") as f :
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                parts.append(page.extract_text())
        return "\n".join(parts)
    except Exception as e:
        return f"Erreur lors du chargement du PDF : {e}"

def load_docx(path: Path) -> str:
    """Lit un fichier DOCX et retourne son contenu en texte."""
    try:
        doc = Document(path)
        parts = [para.text for para in doc.paragraphs]
        return "\n".join(parts)
    except Exception as e:
        return f"Erreur lors du chargement du DOCX : {e}"

def load_csv(path: Path) -> str:
    """Lit un fichier CSV et retourne son contenu en texte."""
    try:
        df = pd.read_csv(path)
        return df.to_string()
    except Exception as e:  
        return f"Erreur lors du chargement du CSV : {e}"

def load_wav(path: Path) -> str:
    """Lit un fichier WAV a réaliser plus tard"""
    return "Contenu du WAV fonctionnalité à implémenter"

def load_wedp(path: Path) -> str:
    """Lit un fichier WEDP et retourne son contenu en texte."""
    try: 
        return "Contenu du WEDP fonctionnalité à implémenter"
    except Exception as e:
        return f"Erreur lors du chargement du WEDP : {e}"

def load_image(path: Path) -> str:
    """Lit un fichier image a réaliser plus tard"""
    return "Contenu de l'image fonctionnalité à implémenter"

DISPATCHE = {
    '.pdf': load_pdf,
    '.docx': load_docx,
    '.csv': load_csv,
    '.wav': load_wav,
    '.wedp': load_wedp,
    '.jpg': load_image,
    '.jpeg': load_image,
    '.png': load_image,
}

def load_documents_from_dir(directory: str) -> Dict[str, str]:
    """ parcours tout le répertoire et charge les fichiers """

    directory = Path(directory)
    if not directory.exists() :
        raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
    
    documents: Dict[str, str] = {}
    for path in directory.rglob('*'):
        if not path.is_file():
            continue
        if path.is_file():
            ext = path.suffix.lower()
            loader = DISPATCHE.get(ext)
            if loader:
                content = loader(path)
                documents[str(path)] = content
            else:
                documents[str(path)] = f"Format de fichier non supporté : {ext}"
    return documents

def main():
    docs = load_documents_from_dir('inputs')
    for filename, content in docs.items():
        print(f"Fichier: {filename}\nContenu:\n{content[:200]}...\n")

if __name__ == "__main__":
    main()