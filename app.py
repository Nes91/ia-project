import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="Menu principal",
        options=["Explication du code", "Projet"],
        default_index=0,
    )
    if selected == "Explication du code":
        st.title(f"Bienvenue dans l'{selected}")
        st.write("Code pour le chargement des classes ImageNet depuis le fichier texte")
        st.code('''
            CLASSES_FILE = "imagenet_classes.json"
            with open(CLASSES_FILE, 'r') as f:
                imagenet_classes = [line.strip() for line in f.readlines()]
    
            model = models.resnet50(pretrained=True)
            model.eval()
            ''', language='python')
    with (st.expander('Explication du code')):
        st.write("CLASSES_FILE = imagenet_classes.json : Nous devons créer un fichier imagenet_classes.json pour le pré-entrainement de l'image et le mettre à la racine du projet.")
        ("with open(CLASSES_FILE, 'r') as f:imagenet_classes = [line.strip() for line in f.readlines()] : Cette ligne ouvre le fichier imagenet_classes.txt en mode lecture ('r').")
        (
            "imagenet_classes = [line.strip() for line in f.readlines()] : f.readlines() lit toutes les lignes du fichier et les stocke dans une liste, La compréhension de liste [line.strip() for line in f.readlines()] parcourt chaque ligne de cette liste, supprime les espaces blancs de début et de fin (avec strip()) et stocke les résultats dans une nouvelle liste appelée imagenet_classes.")
        st.write("model = models.resnet50(pretrained=True) : Cette ligne charge un modèle pré-entraîné ResNet-50 à partir de la bibliothèque torchvision.models, models.resnet50 est une fonction qui retourne une instance de l'architecture ResNet-50"),
        st.write("pretrained=True signifie que le modèle est initialisé avec des poids pré-entraînés sur le jeu de données ImageNet")
        st.write("le modèle ResNet-50 est un réseau de neurones convolutif (Un réseau de neurones convolutif (CNN, pour Convolutional Neural Network) est un type de réseau de neurones artificiels principalement utilisé pour analyser des images) de 50 couches qui est souvent utilisé pour des tâches de classification d'images en raison de sa haute performance.")
        ("model.eval() : Cette ligne met le modèle en mode évaluation.")

    st.write("Code pour la transformation de l'image")
    st.code('''
            preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ''', language='python')
    with st.expander("Explication du code"):
        st.write("transforms.Resize(256) : Cette transformation redimensionne l'image de manière à ce que le plus petit côté soit de 256 pixels tout en maintenant le ratio d'aspect de l'image.")
        st.write("transforms.CenterCrop(224) : Cette transformation découpe une région centrale de l'image de 224x224 pixels.")
        st.write("transforms.ToTensor() : Cette transformation convertit l'image en un tenseur PyTorch. Elle réorganise également les canaux de couleur de l'image (de PIL image format [H x W x C] à PyTorch tensor format [C x H x W]).")
        st.write("transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) : Cette transformation normalise les valeurs de pixels de l'image en utilisant les moyennes et les écarts-types spécifiés pour chaque canal de couleur (RGB). Moyenne : [0.485, 0.456, 0.406], Écart-type : [0.229, 0.224, 0.225]")

    st.write("Code pour la transformation de l'image")
    st.code('''
                def predict(image):
        img_tensor = preprocess(image)
        img_tensor = img_tensor.unsqueeze(0)  # Ajouter une dimension batch
        with torch.no_grad():
            outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        return imagenet_classes[predicted[0]]
        ''', language='python')
    with st.expander("Explication du code"):
        st.write("img_tensor = preprocess(image) : Cette ligne applique les transformations définies par preprocess à l'image d'entrée.")
        st.write("img_tensor = img_tensor.unsqueeze(0) : Cette ligne ajoute une dimension supplémentaire au tenseur d'image.")
        st.write("with torch.no_grad(): : Cette ligne commence un bloc de code dans lequel les gradients ne seront pas calculés.")
        st.write("outputs = model(img_tensor) : Cette ligne passe le tenseur d'image prétraité à travers le modèle pour obtenir les sorties.")
        st.write("_, predicted = outputs.max(1) : Cette ligne trouve l'indice de la classe avec la probabilité la plus élevée")
        st.write("return imagenet_classes[predicted[0]] : Cette ligne utilise l'indice de la classe prédite pour obtenir le nom de la classe à partir de la liste imagenet_classes.")
    st.write("Code pour la transformation de l'image")
    st.code('''
    def load_image(image_file):
        img = Image.open(image_file)
        return img
            ''', language='python')
    with st.expander("Explication du code"):
        st.write("def load_image(image_file): : Cette ligne définit une fonction nommée load_image qui prend un argument image_file.")
        st.write("img = Image.open(image_file) : Cette ligne utilise la bibliothèque PIL (Python Imaging Library) pour ouvrir le fichier d'image")
        st.write("return img : Cette ligne retourne l'objet image ouvert.")
    st.write("Code pour la génération des étiquettes aléatoires pour simuler des prédictions")
    st.code('''
        def generate_random_labels(num_samples):
        
        true_labels = np.random.choice(imagenet_classes, size=num_samples, replace=True)
        predicted_labels = np.random.choice(imagenet_classes, size=num_samples, replace=True)
        return true_labels, predicted_labels
    ''')
    with st.expander("Explication du code"):
        st.write("def generate_random_labels(num_samples) : Cette ligne définit une fonction nommée generate_random_labels qui prend un argument num_samples.")
        st.write("true_labels = np.random.choice(imagenet_classes, size=num_samples, replace=True) : Cette ligne génère une liste d'étiquettes réelles aléatoires.")
        st.write("predicted_labels = np.random.choice(imagenet_classes, size=num_samples, replace=True) : Cette ligne génère une liste d'étiquettes prédites aléatoires")
        st.write("return true_labels, predicted_labels : Cette ligne retourne les listes d'étiquettes réelles et prédites.")
        
    
if selected == "Projet":
    st.title(f"Bienvenue dans la partie {selected}")
    # Chemin du fichier texte contenant les classes ImageNet
    CLASSES_FILE = "imagenet_classes.json"

    # Chargement des classes ImageNet depuis le fichier texte
    with open(CLASSES_FILE, 'r') as f:
        imagenet_classes = [line.strip() for line in f.readlines()]

    model = models.resnet50(pretrained=True)
    model.eval()

    # Transformation de l'image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    def predict(image):
        img_tensor = preprocess(image)
        img_tensor = img_tensor.unsqueeze(0)  # Ajouter une dimension batch
        with torch.no_grad():
            outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        return imagenet_classes[predicted[0]]


    def load_image(image_file):
        img = Image.open(image_file)
        return img


    def generate_random_labels(num_samples):
        # Générer des étiquettes aléatoires pour simuler des prédictions
        true_labels = np.random.choice(imagenet_classes, size=num_samples, replace=True)
        predicted_labels = np.random.choice(imagenet_classes, size=num_samples, replace=True)
        return true_labels, predicted_labels



    st.title("Reconnaissance d'Image avec Streamlit")

    image_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        img = load_image(image_file)
        st.image(img, caption='Image téléchargée.', use_column_width=True)

        if st.button('Reconnaître l\'image'):
            label = predict(img)
            st.write(f'La classe prédite est : {label}')

            # Afficher les options avancées dans la sidebar
            st.sidebar.title("Options avancées")

            # show_confusion_matrix = st.sidebar.checkbox('Afficher la matrice de confusion')
            # show_improvement_suggestions = st.sidebar.checkbox('Afficher les axes d\'amélioration')



        if st.checkbox("Afficher les axes d'améliorations"):

            st.write("""
            Axes d'amélioration des résultats
            1. **Amélioration des données** :
               - Collecter plus de données d'entraînement de haute qualité.
               - Augmenter la diversité des données d'entraînement.

            2. **Amélioration du modèle** :
               - Entraîner un modèle personnalisé sur vos propres données.
               - Essayer des architectures de modèles plus récentes et performantes.

            3. **Optimisation des hyperparamètres** :
               - Ajuster les taux d'apprentissage, les tailles de batch, etc.

            4. **Techniques de régularisation** :
               - Utiliser des techniques comme le dropout, la normalisation batch, etc.

            5. **Post-traitement des résultats** :
               - Affiner les résultats avec des techniques de post-traitement comme la suppression des détections non valides.
            """)

        if st.checkbox("### Matrice de confusion"):
            st.write("### Matrice de confusion")

            # Générer des étiquettes aléatoires pour la matrice de confusion
            num_samples = 100
            true_labels, predicted_labels = generate_random_labels(num_samples)

            # Calculer la matrice de confusion
            cm = confusion_matrix(true_labels, predicted_labels, labels=imagenet_classes[:num_samples])

            # Afficher la matrice de confusion
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cm, annot=False, fmt="d", cmap="YlGnBu", xticklabels=True, yticklabels=True, ax=ax)
            plt.xlabel('Prédictions')
            plt.ylabel('Vérité Terrain')
            st.pyplot(fig)
