import numpy as np
import cv2 as cv
import os

# Charger les images
dossier = "Photo/Photo objet 7"
fichiers=[]
nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)


def reshape (img,scale_percent) :
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized_img

# --- PARAMÈTRES ---
image_path = r"C:\Users\phoat\Desktop\photogrammetrie\ballon.png"  # Chemin de l'image
max_points = 150 # Nombre maximal de points détectés
min_distance = 25  # Distance minimale entre les points détectés
quality_level = 0.01  # Qualité des points détectés
scale_factor = 0.20 # Facteur de zoom initial
min_zoom, max_zoom = 0.5, 3.0  # Limites du zoom

# --- CHARGEMENT DE L'IMAGE ---
original_image = cv.imread(fichiers[0])
image = original_image.copy()

# Liste des points détectés
points = []
mode = "suppression"  # Mode actif: "ajout" ou "suppression"

def pointage(img):
    """Détecte les coins et les affiche avec un indice."""
    global points

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Détection des coins
    corners = cv.goodFeaturesToTrack(gray, maxCorners=max_points, qualityLevel=quality_level, minDistance=min_distance)
    if corners is not None:
        corners = np.int32(corners)  # Convertir en entiers
        points = corners.reshape(-1, 2).tolist()

    print("\n📍 Points détectés au démarrage :", points)
    return afficher_points()

def afficher_points():
    """Affiche les points détectés et met à jour les boutons."""
    global scale_factor

    img = cv.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
    
    for i, (x, y) in enumerate(points):
        x_zoomed = int(x * scale_factor)
        y_zoomed = int(y * scale_factor)
        cv.circle(img, (x_zoomed, y_zoomed), 5, (0, 0, 255), -1)  
        cv.putText(img, str(i), (x_zoomed + 5, y_zoomed - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    h, w, _ = img.shape
    button_width, button_height = 50, 30
    x_add, y_add = w - 120, h - 40
    x_rem, y_rem = w - 60, h - 40

    color_add = (0, 255, 0) if mode == "ajout" else (150, 150, 150)
    color_rem = (0, 0, 255) if mode == "suppression" else (150, 150, 150)

    cv.rectangle(img, (x_add, y_add), (x_add + button_width, y_add + button_height), color_add, -1)
    cv.putText(img, "+", (x_add + 15, y_add + 22), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.rectangle(img, (x_rem, y_rem), (x_rem + button_width, y_rem + button_height), color_rem, -1)
    cv.putText(img, "X", (x_rem + 15, y_rem + 22), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    return img

def on_mouse(event, x, y, flags, param):
    """Gère les événements de la souris : clics et molette pour zoom."""
    global mode, scale_factor

    img_display = afficher_points()
    h, w, _ = img_display.shape

    button_width, button_height = 50, 30
    x_add, y_add = w - 120, h - 40
    x_rem, y_rem = w - 60, h - 40

    if event == cv.EVENT_LBUTTONDOWN:
        if x_add <= x <= x_add + button_width and y_add <= y <= y_add + button_height:
            mode = "ajout"
            print("🔵 Mode Ajout activé")
        elif x_rem <= x <= x_rem + button_width and y_rem <= y <= y_rem + button_height:
            mode = "suppression"
            print("🔴 Mode Suppression activé")
        else:
            x_original = int(x / scale_factor)
            y_original = int(y / scale_factor)
            if mode == "suppression":
                supprimer_point(x_original, y_original)
            elif mode == "ajout":
                ajouter_point(x_original, y_original)

        img_update = afficher_points()
        cv.imshow('Interactive Points', img_update)

    elif event == cv.EVENT_MOUSEWHEEL:
        zoom_step = 0.1
        if flags > 0:
            scale_factor = min(scale_factor + zoom_step, max_zoom)
        else:
            scale_factor = max(scale_factor - zoom_step, min_zoom)

        img_update = afficher_points()
        cv.imshow('Interactive Points', img_update)

def supprimer_point(x_click, y_click):
    """Supprime un point et met à jour la liste des indices."""
    global points

    for i, (x, y) in enumerate(points):
        if abs(x - x_click) < 10 and abs(y - y_click) < 10:
            print("\n📌 Point supprimé :", (x, y))
            points.pop(i)
            print("📜 Liste mise à jour :", points)
            break

    img_update = afficher_points()
    cv.imshow('Interactive Points', img_update)

def ajouter_point(x_click, y_click):
    """Ajoute un nouveau point à la liste si la limite n'est pas atteinte."""
    global points
    points.append([x_click, y_click])
    print("\n➕ Point ajouté :", (x_click, y_click))
    print("📜 Liste mise à jour :", points)
    
    img_update = afficher_points()
    cv.imshow('Interactive Points', img_update)

image_annotated = pointage(image)
cv.imshow('Interactive Points', image_annotated)
cv.setMouseCallback('Interactive Points', on_mouse)
cv.waitKey(0)
cv.destroyAllWindows()
