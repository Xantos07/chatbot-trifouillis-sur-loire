# Étape 1 : Chargement des données (Loading)

import PyPDF2 
from docx import Document
import pandas as pd

def load_pdf():
    return "Contenu du PDF"

def load_docx():
    return "Contenu du DOCX"

def load_csv():
    return "Contenu du CSV"

def load_wav():
    return "Contenu du WAV"

def load_wedp():
    return "Contenu du WEDP"

def load_image():
    return "Contenu de l'image"


def load_documents (type):

    loaders = {
        'pdf': load_pdf,
        'docx': load_docx,
        'csv': load_csv,
        'wav': load_wav,
        'wedp': load_wedp,
        'image': load_image,
    }

    loader = loaders.get(type)
    if loaders:
        return loader()
    else:
        raise ValueError(f"Type de document inconnu : {type}")
