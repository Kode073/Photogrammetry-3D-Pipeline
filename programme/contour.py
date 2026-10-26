import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

# Charger les images
dossier = "Photo/Photo objet 6"
fichiers=[]
nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)


def reshape (img,scale_percent) :
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

image = fichiers[1]

def detect_contours_and_vertices(image_path):
    """ Détecte les contours de l'objet et extrait les sommets de sa forme polygonale. """
    
    # 1️. Charger l'image et convertir en niveaux de gris
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️. Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3️. Détection des contours avec Canny
    edges = cv2.Canny(blurred, 50, 150)

    # 4️. Fermer les petits trous avec une transformation morphologique
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5️. Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6️. Trier pour garder le plus grand contour (l'objet principal)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if not contours:
        print("❌ Aucun contour détecté !")
        return

    main_contour = contours[0]

    # 7️. Approximation polygonale pour extraire les sommets
    epsilon = 0.02 * cv2.arcLength(main_contour, True)  # Ajuste la précision
    approx = cv2.approxPolyDP(main_contour, epsilon, True)

    # 8️. Dessiner les contours et les sommets détectés
    result = img.copy()
    cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)  # Dessine le polygone

    # Dessiner les sommets détectés
    for point in approx:
        x, y = point.ravel()
        cv2.circle(result, (x, y), 5, (0, 0, 255), -1)  # Marquer les sommets en rouge

    # 9️. Afficher le résultat
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Contours et sommets détectés")
    plt.axis("off")
    plt.show()

    return approx  # Retourne la liste des sommets

detect_contours_and_vertices(image)