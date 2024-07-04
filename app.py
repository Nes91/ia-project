import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

with st.sidebar:
    selected = option_menu(
        menu_title="Menu principal",
        options=["Explication du code", "Projet"],
        default_index=0,
    )
if selected == "Explication du code":
    st.title(f"Bienvenue dans l'{selected}")
    
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
