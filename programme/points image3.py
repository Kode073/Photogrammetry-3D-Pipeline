import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from SuperGluePretrainedNetwork.models.matching import Matching


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matching = Matching({})
matching = matching.to(device)

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

    


def matches (image1, image2) :
    img0 = reshape(cv2.imread(image1,cv2.IMREAD_GRAYSCALE),50)
    img1 = reshape(cv2.imread(image2,cv2.IMREAD_GRAYSCALE),50)

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
    coords0 = [ (int(kp.pt[0]), int(kp.pt[1])) for kp in kp0 ]
    coords1 = [ (int(kp.pt[0]), int(kp.pt[1])) for kp in kp1 ]

    return coords0,coords1,resized_img




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


fichiers = liste_photo(fichiers)
print(fichiers)
dict = points(fichiers,2)
print(len(dict))

























    
