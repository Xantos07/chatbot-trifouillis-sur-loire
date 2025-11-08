import subprocess
import os

def list_files(directory):
    return os.listdir(directory)

def lister_files(directory):
    for racine, repertoires, fichiers in os.walk(directory):
        for fichier in fichiers:
            print(os.path.join(racine, fichier))

def convert_all_file_to_markdown(directory):
    if(not os.path.exists('outputs')):
        os.makedirs('outputs')

    supported_ext = ('.pdf', '.docx', '.doc', '.odt', '.txt')

    for racine, repertoires, fichiers in os.walk(directory):
        for fichier in fichiers:
            if not fichier.lower().endswith(supported_ext):
                print(f"Le fichier {fichier} n'est pas dans un format supporté pour la conversion.")
                continue
            convert_to_markdown(os.path.join(racine, fichier), os.path.join('outputs', fichier + '.md'))

#convert file to markdown   
def convert_to_markdown(input_path, output_path):
    cmd = ['docling', input_path, '--to', 'md', '--output', output_path]
    print(f"Exécution de la commande : {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print("Conversion réussie")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la conversion de {input_path} :")
        print(e.stderr)



print(lister_files('inputs'))
convert_all_file_to_markdown('inputs')