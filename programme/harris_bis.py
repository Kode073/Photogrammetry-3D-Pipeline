import numpy as np
import cv2 as cv
import os

# Charger les images
dossier = "Photo/Photo objet 7"
fichiers=[]

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)

# --- PARAMÈTRES ---
image_path = r"C:\Users\phoat\Desktop\photogrammetrie\ballon.png"
max_points = 6
min_distance = 25
quality_level = 0.01
scale_factor = 0.2  # Facteur de zoom initial
min_zoom, max_zoom = 0.5, 10.0  # Limites du zoom
zone_clicks_required = 6

# --- CHARGEMENT DE L'IMAGE ---
original_image = cv.imread(fichiers[0])
image = original_image.copy()
h, w = original_image.shape[:2]  # Taille originale

# Liste des points détectés
points = []
zone_points = []
mode = "ajout"
zone_closed = False

# Interface graphique (suppression du bouton "Zone")
BUTTONS = {"Ajout": (20, 20), "Suppression": (120, 20)}
button_size = (90, 30)

def draw_buttons(img):
    for text, (x, y) in BUTTONS.items():
        color = (0, 255, 0) if mode.lower() in text.lower() else (200, 200, 200)
        cv.rectangle(img, (x, y), (x + button_size[0], y + button_size[1]), color, -1)
        cv.putText(img, text, (x + 10, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def check_button_click(x, y):
    global mode
    for text, (bx, by) in BUTTONS.items():
        if bx <= x <= bx + button_size[0] and by <= y <= by + button_size[1]:
            mode = text.lower()
            print(f"🔘 Mode activé: {mode}")
            return True
    return False

def pointage():
    """Détecte les coins dans la zone définie."""
    global points
    if len(zone_points) < zone_clicks_required:
        return
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.fillPoly(mask, [np.array(zone_points)], 255)
    
    gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    corners = cv.goodFeaturesToTrack(gray, maxCorners=max_points, qualityLevel=quality_level, minDistance=min_distance, mask=mask)
    
    if corners is not None:
        points = np.int32(corners).reshape(-1, 2).tolist()
    
    print("\n📍 Points détectés :", points)
    afficher_points()

def afficher_points():
    """Affiche l'image avec les points et la zone sélectionnée."""
    img = cv.resize(original_image, (int(w * scale_factor), int(h * scale_factor)))
    draw_buttons(img)

    # Afficher la zone
    if len(zone_points) > 1:
        scaled_zone = [(int(x * scale_factor), int(y * scale_factor)) for x, y in zone_points]
        cv.polylines(img, [np.array(scaled_zone)], isClosed=zone_closed, color=(255, 255, 0), thickness=2)

    # Afficher les points détectés
    for i, (x, y) in enumerate(points):
        sx, sy = int(x * scale_factor), int(y * scale_factor)
        cv.circle(img, (sx, sy), 5, (0, 0, 255), -1)
        cv.putText(img, str(i), (sx + 5, sy - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    cv.imshow('Interactive Points', img)

def on_mouse(event, x, y, flags, param):
    """Gestion des clics et du zoom."""
    global mode, scale_factor, zone_points, zone_closed

    if event == cv.EVENT_LBUTTONDOWN:
        if check_button_click(x, y):
            afficher_points()
            return

        x_original = int(x / scale_factor)
        y_original = int(y / scale_factor)

        if len(zone_points) < zone_clicks_required:
            zone_points.append((x_original, y_original))
            print("🟡 Point ajouté à la zone :", zone_points)
            if len(zone_points) == zone_clicks_required:
                zone_closed = True
                print("✅ Zone complète, lancement de la détection")
                pointage()
        elif mode == "suppression":
            supprimer_point(x_original, y_original)
        elif mode == "ajout":
            ajouter_point(x_original, y_original)

    elif event == cv.EVENT_MOUSEWHEEL:
        zoom_step = 0.1
        if flags > 0:
            scale_factor = min(scale_factor + zoom_step, max_zoom)
        else:
            scale_factor = max(scale_factor - zoom_step, min_zoom)

    afficher_points()

def supprimer_point(x_click, y_click):
    """Supprime un point proche du clic."""
    global points
    points = [p for p in points if not (abs(p[0] - x_click) < 10 and abs(p[1] - y_click) < 10)]
    print("📌 Point supprimé, nouvelle liste :", points)
    afficher_points()

def ajouter_point(x_click, y_click):
    """Ajoute un point manuel."""
    global points
    points.append([x_click, y_click])
    print("➕ Point ajouté, nouvelle liste :", points)
    afficher_points()

cv.imshow('Interactive Points', original_image)
cv.setMouseCallback('Interactive Points', on_mouse)
afficher_points()
cv.waitKey(0)
cv.destroyAllWindows()
