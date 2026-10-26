import cv2
import numpy as np
import os

# Charger les images
dossier = "Photo/Photo code gray"
fichiers=[]

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)


img1 = cv2.imread(fichiers[0])
img2 = cv2.imread(fichiers[1])

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Détecter et décrire les features avec SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Configurer FLANN
index_params = dict(algorithm=1, trees=100)  # Algorithme KDTree
search_params = dict(checks=50)  # Nombre d'itérations pour trouver les correspondances

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)


# Appliquer le ratio test de Lowe
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)




# Affichage d'une image redimensionnée
scale_percent = 20  # Réduction à 50% de la taille d'origine
width = int(matched_img.shape[1] * scale_percent / 100)
height = int(matched_img.shape[0] * scale_percent / 100)
dim = (width, height)

resized_img = cv2.resize(matched_img, dim, interpolation=cv2.INTER_AREA)


cv2.imshow('Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()





import numpy as np

# Récupérer les coordonnées et descripteurs des bons matchs
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

descriptors1_matched = np.array([descriptors1[m.queryIdx] for m in good_matches])
descriptors2_matched = np.array([descriptors2[m.trainIdx] for m in good_matches])

# Afficher quelques exemples de coordonnées et descripteurs
print(len(points1),len(points2),len(keypoints1))
#print("Coordonnées des points dans l'image 1 :\n", points1[:5])
#print("Coordonnées des points dans l'image 2 :\n", points2[:5])

#print("Descripteurs des points dans l'image 1 :\n", descriptors1_matched[:5])
#print("Descripteurs des points dans l'image 2 :\n", descriptors2_matched[:5])

