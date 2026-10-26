import os
import cv2
import numpy as np
import math

path = "Photo/Photo_croix/image1.png"
image = cv2.imread(path)
liste_points = [(278, 147), (332, 277), (112, 183), (431, 117), (229, 63)]



def coor_polaire (center,p) :
    r = np.sqrt((p[0]-center[0])**2+(p[1]-center[1])**2)
    theta = math.atan2(p[1] - center[1], p[0] - center[0])
    return [r,theta]



def numero (img,L) :
    l = []
    polaire = []
    w = int(img.shape[1])
    h = int(img.shape[0])
    centre = [h/2,w/2]
    for i in range (len(L)) :
        polaire.append((coor_polaire(centre,L[i]),i))
        pollaire_tries = sorted(polaire, key=lambda p: p[0][1], reverse=True)
    for el in pollaire_tries :
        l.append(L[el[1]])
    return l

sorted_points = numero(image,liste_points)
for i, point in enumerate(sorted_points):
    cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)
    cv2.putText(image, str(i+1), tuple(point + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

print(sorted_points)
# Affichage (si tu utilises un environnement graphique)
cv2.imshow("Points numérotés", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

    





