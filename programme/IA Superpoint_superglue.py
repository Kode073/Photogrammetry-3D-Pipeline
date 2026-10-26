import torch
import cv2
import numpy as np
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from SuperGluePretrainedNetwork.models.matching import Matching
import os

#{'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024} pour superpoints
#{'superpoint':{'sinkhorn_iterations': 200,'match_threshold': 0.2,}}

def reshape (img,scale_percent) :
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img


# Charger les images
dossier = "video photo"
fichiers=[]

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)

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

fichiers = liste_photo(fichiers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matching = Matching({})
matching = matching.to(device)


img0 = reshape(cv2.imread(fichiers[0],cv2.IMREAD_GRAYSCALE),100)
img1 = reshape(cv2.imread(fichiers[32],cv2.IMREAD_GRAYSCALE),100)

imgsuper0 = img0.astype('float32') / 255.0
imgsuper0 = torch.from_numpy(imgsuper0).unsqueeze(0).unsqueeze(0).to(device)

imgsuper1 = img1.astype('float32') / 255.0
imgsuper1= torch.from_numpy(imgsuper1).unsqueeze(0).unsqueeze(0).to(device)


data = {'image0':imgsuper0, 'image1':imgsuper1}
data['image0'] = data['image0'].to(device)  # Déplace l'image sur le même dispositif
data['image1'] = data['image1'].to(device)  # Déplace l'image sur le même dispositif

pred = matching(data)

matches = pred['matches0'][0]

# Conversion correcte des keypoints en tableaux NumPy
keypoints0 = pred['keypoints0'][0]
keypoints1 = pred['keypoints1'][0]

confidences0 = pred['matching_scores0'][0]
confidence_threshold = 0.90
valid_matches_confidence = (confidences0 > confidence_threshold) 

# Vérification : Afficher le nombre de correspondances valides
print(f"Nombre de correspondances valides : {valid_matches_confidence.sum()}")


# Création des objets KeyPoint pour OpenCV
kp0 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints0[valid_matches_confidence]]
kp1 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints1[matches[valid_matches_confidence]]]

matches_cv = [cv2.DMatch(i, i, 0) for i in range(valid_matches_confidence.sum())]

# Dessiner les correspondances avec OpenCV
img_matches = cv2.drawMatches(img0, kp0, img1, kp1, matches_cv, None)


resized_img = reshape(img_matches,100)

cv2.imshow('Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
