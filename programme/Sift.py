import cv2
import numpy as np
import os

# Charger les images
dossier = "Photo/Photo objet 7"
fichiers=[]

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)

print(fichiers)
img1 = cv2.imread(fichiers[0])

sift = cv2.SIFT_create(150)
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)

# draw the detected key points
sift_image = cv2.drawKeypoints(img1, keypoints1, img1)



# Affichage d'une image redimensionnée
scale_percent = 50  # Réduction à 50% de la taille d'origine
width = int(sift_image.shape[1] * scale_percent / 100)
height = int(sift_image.shape[0] * scale_percent / 100)
dim = (width, height)

resized_img = cv2.resize(sift_image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

