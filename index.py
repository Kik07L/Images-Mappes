import os
import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Chemin vers le dossier contenant les images
dossier_images = "img"

# Charger toutes les images du dossier et les redimensionner en 100x100 pixels
images = []
for nom_fichier in os.listdir(dossier_images):
    chemin_image = os.path.join(dossier_images, nom_fichier)
    if os.path.isfile(chemin_image):
        image = cv2.imread(chemin_image)
        image = cv2.resize(image, (340, 160))  # Redimensionner en 100x100
        images.append(image)
        print("Redimensionnement de l'image :", nom_fichier)

# Convertir les images en caractéristiques uniques
features = np.array([image.flatten() for image in images])

# Appliquer t-SNE pour réduire la dimensionnalité des caractéristiques en 2D
tsne = TSNE(n_components=2, random_state=10)
tsne_result = tsne.fit_transform(features)

# Créer une figure pour afficher les images
fig, ax = plt.subplots()

# Parcourir les résultats t-SNE et afficher les images
for image, (x, y) in zip(images, tsne_result):
    # Afficher l'image à la position (x, y) dans la figure
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), extent=(x-0.5, x+0.5, y-0.5, y+0.5))
    ax.set_xlim(tsne_result[:, 0].min()-1, tsne_result[:, 0].max()+1)
    ax.set_ylim(tsne_result[:, 1].min()-1, tsne_result[:, 1].max()+1)
    ax.axis('off')

# Afficher la figure avec les images t-SNE
plt.show()