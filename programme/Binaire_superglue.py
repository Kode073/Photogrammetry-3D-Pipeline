import torch
import cv2
import numpy as np
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
import os

superglue = SuperGlue({})

img1 = []
img2 =[]
points1 = []
points2 = []

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
        print(coord)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(img, keypoints)
    return keypoints,descriptors

keypoints1, descriptors1 = creation_descriptors(img1,points1)
keypoints2, descriptors2 = creation_descriptors(img2,points2) 

data ={}
data['descriptors0'] = descriptors1
data['descriptors1'] = descriptors2
data['keypoints0'] = keypoints1
data['keypoints1'] = keypoints2

pred = superglue(data)

matches = pred['matches0'][0]

# Conversion correcte des keypoints en tableaux NumPy
keypoints_glue0 = pred['keypoints0'][0]
keypoints_glue1 = pred['keypoints1'][0]

confidences0 = pred['matching_scores0'][0]
confidence_threshold = 0.86
valid_matches_confidence = (confidences0 > confidence_threshold) 

# Vérification : Afficher le nombre de correspondances valides
print(f"Nombre de correspondances valides : {valid_matches_confidence.sum()}")

# Création des objets KeyPoint pour OpenCV
kp1 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints_glue0[valid_matches_confidence]]
kp2 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints_glue1[matches[valid_matches_confidence]]]

matches_cv = [cv2.DMatch(i, i, 0) for i in range(valid_matches_confidence.sum())]

# Dessiner les correspondances avec OpenCV
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches_cv, None)


resized_img = reshape(img_matches,100)

cv2.imshow('Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



