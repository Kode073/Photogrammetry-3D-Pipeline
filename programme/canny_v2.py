import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Charger les images
dossier = "video photo"
fichiers=[]
nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)


def reshape (img,scale_percent) :
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

def detect_contours_with_roi(image_path, crop_ratio=0.5):
    """Détecte les contours uniquement dans une zone centrale de l'image."""

    # 1️⃣ Charger l’image et convertir en niveaux de gris
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Obtenir les dimensions de l’image
    h, w = gray.shape

    # 3️⃣ Définir la zone centrale
    x1, y1 = int(w * (1 - crop_ratio) / 2), int(h * (1 - crop_ratio) / 2)
    x2, y2 = int(w * (1 + crop_ratio) / 2), int(h * (1 + crop_ratio) / 2)

    # 4️⃣ Créer un masque noir et y dessiner un rectangle blanc au centre
    mask = np.zeros_like(gray)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # 5️⃣ Appliquer le masque pour ne garder que la zone centrale
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # 6️⃣ Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)

    # 7️⃣ Détection des contours avec Canny
    edges = cv2.Canny(blurred, 50, 150)

    # 8️⃣ Fermer les petits trous avec une transformation morphologique
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 9️⃣ Appliquer le masque aux contours
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # 🔟 Trouver les contours uniquement dans la zone centrale
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ Aucun contour détecté !")
        return

    # 🔹 Trier pour garder le plus grand contour (l'objet principal)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    main_contour = contours[0]

    # 🔹 Approximation polygonale pour extraire les sommets
    epsilon = 0.02 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)

    # 🔹 Dessiner le contour détecté
    result = img.copy()
    cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)  # Polygone en vert

    # 🔹 Dessiner les sommets détectés
    for point in approx:
        x, y = point.ravel()
        cv2.circle(result, (x, y), 5, (0, 0, 255), -1)  # Sommets en rouge

     # 🔹 Ajouter un cadre bleu pour visualiser la zone centrale
    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Cadre bleu fin

    # 🔹 Afficher l’image avec les contours
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Contours détectés (zone centrale uniquement)")
    plt.axis("off")
    plt.show()

    return approx  # Retourne les sommets détectés

# 🔹 Appliquer la détection sur une image
vertices = detect_contours_with_roi(fichiers[30], crop_ratio=0.6)
