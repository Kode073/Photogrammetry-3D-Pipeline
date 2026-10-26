import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

dossier = "video photo"
fichiers=[]

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)

def reshape (img,scale_percent) :
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img

def creation_descriptors (image,points) :
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    keypoints = []
    for coord in points :
        keypoint = cv2.KeyPoint(x=np.float32(coord[0][0]),y=np.float32(coord[0][1]), size=20)
        keypoints.append(keypoint)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(img, keypoints)
    return keypoints,descriptors
    


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
    #plt.figure(figsize=(10, 5))
    #plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #plt.title("Contours détectés (zone centrale uniquement)")
    #plt.axis("off")
    #plt.show()

    return approx  # Retourne les sommets détectés


def matches (image1, image2) :
    vertices1 = detect_contours_with_roi(image1)
    vertices2 = detect_contours_with_roi(image2)
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    keypoints1, descriptors1 = creation_descriptors(image1,vertices1)
    keypoints2, descriptors2 = creation_descriptors(image2,vertices2)
    # Configurer FLANN
    index_params = dict(algorithm=2, trees=100)  # Algorithme KDTree
    search_params = dict(checks=100)  # Nombre d'itérations pour trouver les correspondances

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Appliquer le ratio test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)


    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    return pts1,pts2,matched_img




def print_matches(image1, image2) :
    _,_,matched_img = matches(image1,image2)
    cv2.imshow('Image', reshape(matched_img,25))
    cv2.waitKey(0)
    cv2.destroyAllWindows()





####################################################################################################################################
def transfo (x,t) :
    c = np.append(x,t)
    return c

#transforme une liste de coordonnée euclidien 
def transfo_l (l,ti) : 
    l2 = []
    for i in range (len(l)) :
        l2.append(transfo(l[i],int(ti)))
    return l2

def in_dict(a,b) :
    for el in b :
        if (a==el).all() :
            return True
    return False

def rajout (d,points1,points2,ind1,ind2) :
    points1 = transfo_l(points1,ind1)
    points2 = transfo_l(points2,ind2)
    for i in range (len(points1)) :
        in_d = False
        for el in d :
            if in_dict(points1[i], d[el]) :
                in_d = True
                if not in_dict(points2[i], d[el]) :
                   d[el].append(points2[i])
        if not in_d :
            d[len(d)+1] = [points1[i],points2[i]]

    for i in range (len(points2)) :
        in_d = False
        for el in d :
            if in_dict(points2[i], d[el]) :
                in_d = True
                if not in_dict(points1[i], d[el]):
                   d[el].append(points1[i])
        if not in_d :
            d[len(d)+1] = [points1[i],points2[i]]

    return d



def points (fichiers,r) :
    d = {}
    n = len(fichiers)
    for i in range (n) :
        for j in range (i+1,min(i+r+1,n)) :
            print(i,j)
            points1,points2,matched_img= matches(fichiers[i],fichiers[j])
            d = rajout(d,points1,points2,i,j)
    return d



def liste_photo(fichiers) :
    l = []
    for ind,nom in enumerate (fichiers) :
        nom2 = nom.split("/")
        nom3 = nom2[1].split(".")
        l.append((nom3[0],ind))
    l = sorted(l,key=lambda x : int(x[0]))
    photos = []
    for el in l :
        photos.append(fichiers[int(el[1])])
    return photos

def liste_coord(fichiers,r,l) :
    L = [[] for i in range (len(l))]
    d = points(fichiers,r)
    print(d)
    for key in d :
        for i in range (len(L)) :
            in_l = False
            for el in d[key] :
                if el[2] == l[i] :
                    in_l = True
                    L[i].append((int(el[0]),int(el[1])))
            if not in_l :
                L[i].append(None)
    bon = False
    while bon == False :
        for j in range (len(L)) :
            if L[j][-1] != None :
                bon  = True
        if bon == False :
            for j in range (len(L)) :
                L[j].pop(-1)
    return L
            


fichiers = liste_photo(fichiers)
print(liste_coord(fichiers,2,[0,9,14,33]))
#print(fichiers)
#print(points(fichiers,2))

























    
