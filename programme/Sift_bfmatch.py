import cv2 
import os

dossier = "Photo/Photo objet 1"
fichiers=[]

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)

#dossier = "image"
#nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
#for f in os.listdir(dossier) :
#    img = cv2.imread(dossier+"/"+f)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('photo',img)
#cv2.waitKey(0)


dossier = "Photo/Photo objet 1"
fichiers=[]

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)


img1 = cv2.imread(fichiers[0])
img2 = cv2.imread(fichiers[1])
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors1,descriptors2)
matches = sorted(matches, key = lambda x:x.distance)
# draw first 50 matches
matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, img2, flags=2)
#cv2.imshow('image', matched_img)
#cv2.waitKey(0)


sift_image = cv2.drawKeypoints(img1, keypoints1, img1)
# Affichage d'une image redimensionnée
scale_percent = 50  # Réduction à 50% de la taille d'origine
width = int(sift_image.shape[1] * scale_percent / 100)
height = int(sift_image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_img = cv2.resize(sift_image, dim, interpolation=cv2.INTER_AREA)

keypoints_coords = [kp.pt for kp in keypoints1] 
print(keypoints_coords[:3])
cv2.imshow('image', resized_img)
cv2.waitKey(0)