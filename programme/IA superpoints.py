import torch
import cv2
import numpy as np
import os
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from SuperGluePretrainedNetwork.models.matching import Matching

def reshape (img,scale_percent) :
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

superpoint  = SuperPoint({'max_keypoints': -1})
superpoint = superpoint.to(device)  # Déplace le modèle sur GPU/CPU

# Charger les images
dossier = "Photo/Photo objet 7"
fichiers=[]

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)

img1 = reshape(cv2.imread(fichiers[0],cv2.IMREAD_GRAYSCALE),60)

imgsuper = img1.astype('float32') / 255.0
imgsuper = torch.from_numpy(imgsuper).unsqueeze(0).unsqueeze(0).to(device)


data = {'image':imgsuper}
data['image'] = data['image'].to(device)  # Déplace l'image sur le même dispositif

pred = superpoint(data)

keypoints = pred['keypoints'][0]

for keypoint in keypoints:
    cv2.circle(img1, (int(keypoint[0]), int(keypoint[1])), radius=5, color=(0, 0, 255), thickness=-1)  # Rouge, remplissage plein

scale_percent = 100  # Réduction à 50% de la taille d'origine
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)

resized_img = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)


cv2.imshow('Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(keypoints))