import torch
import cv2
import numpy as np
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
import os

def reshape (img,scale_percent) :
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superpoint  = SuperPoint({'descriptor_dim': 256,'keypoint_threshold': 0.0005})
superpoint = superpoint.to(device)  # Déplace le modèle sur GPU/CPU

# Charger les images
dossier = "Photo/video photo"
fichiers=[]

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)

print(fichiers)
img1 = cv2.imread(fichiers[0])
img2 = cv2.imread(fichiers[1])

img1_reshape = reshape(img1,100)
img2_reshape = reshape(img2,100)

print(img1_reshape.shape)
print(img2_reshape.shape)

imgsuper1 = img1_reshape.astype('float32') / 255.0
imgsuper1 = torch.from_numpy(imgsuper1).unsqueeze(0).unsqueeze(0).to(device)
data1 = {'image':imgsuper1}
data1['image'] = data1['image'].to(device)  # Déplace l'image sur le même dispositif
pred1 = superpoint(data1)
keypoints1 = pred1['keypoints'][0]
descriptors1 = pred1['descriptors'][0]

imgsuper2 = img2_reshape.astype('float32') / 255.0
imgsuper2 = torch.from_numpy(imgsuper2).unsqueeze(0).unsqueeze(0).to(device)
data2 = {'image':imgsuper2}
data2['image'] = data2['image'].to(device)  # Déplace l'image sur le même dispositif
pred2 = superpoint(data2)
keypoints2 = pred2['keypoints'][0]
descriptors2 = pred2['descriptors'][0]

descriptors1 = descriptors1.detach().cpu().numpy().astype(np.float32)
descriptors2 = descriptors2.detach().cpu().numpy().astype(np.float32)
descriptors1 = descriptors1.T  # Transposition
descriptors2 = descriptors2.T

print('fin')
# Configurer FLANN
index_params = dict(algorithm=2, trees=100)  # Algorithme KDTree
search_params = dict(checks=100)  # Nombre d'itérations pour trouver les correspondances

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Appliquer le ratio test de Lowe
good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m)


def to_cvkeypoints(keypoints_array):
    return [cv2.KeyPoint(int(kp[0]), int(kp[1]),1) for kp in keypoints_array]

keypoints1 = to_cvkeypoints(keypoints1)
keypoints2 = to_cvkeypoints(keypoints2)


matched_img = cv2.drawMatches(img1_reshape, keypoints1, img2_reshape, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



resized_img = reshape(matched_img,25)

cv2.imshow('Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



