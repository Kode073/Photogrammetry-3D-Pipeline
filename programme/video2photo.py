import os 
import numpy as np
import cv2

dossier = "video"
fichiers=[]
cnt = 0

nombre_fichiers = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
for f in os.listdir(dossier) :
    fichiers.append(dossier+"/"+f)

div = 10
print(fichiers[1])

capture = cv2.VideoCapture(fichiers[1])
while True : 
    isTrue,frame = capture.read()
    if not isTrue :
       break
    cv2.imshow('Video',frame)
    cnt = cnt+1
    if cnt%div == 0 :
        cv2.imwrite(str(cnt)+'.png', frame)
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

print(cnt)
cv2.destroyAllWindows()
cv2.waitKey(0)

