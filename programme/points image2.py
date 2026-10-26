import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superpoint  = SuperPoint({'descriptor_dim': 256,'keypoint_threshold': 0.0005})
superpoint = superpoint.to(device)  # Déplace le modèle sur GPU/CPU

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

    
def to_cvkeypoints(keypoints_array):
    return [cv2.KeyPoint(int(kp[0]), int(kp[1]),1) for kp in keypoints_array]

def matches (image1, image2) :
    img1 = cv2.imread(image1,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2,cv2.IMREAD_GRAYSCALE)
    img1_reshape = reshape(img1,60)
    img2_reshape = reshape(img2,60)
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

    # Configurer FLANN
    index_params = dict(algorithm=2, trees=50)  # Algorithme KDTree
    search_params = dict(checks=50)  # Nombre d'itérations pour trouver les correspondances

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    print("fin")

    # Appliquer le ratio test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    keypoints1 = to_cvkeypoints(keypoints1)
    keypoints2 = to_cvkeypoints(keypoints2)
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    return pts1,pts2


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
            points1,points2= matches(fichiers[i],fichiers[j])
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


fichiers = liste_photo(fichiers)
print(fichiers)
print(points(fichiers,2))

























    
