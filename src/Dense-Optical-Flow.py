import time

import cv2
import numpy
import numpy as np


"""
Q2

pyr_scale) pas bcp de diff
levels) plus petit = plus instable (tache de couleur sur l'optical flow)
winsize) larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field. 
iterations) flou avant 3 puis plus d'amélioration
poly_n) flou avant 5 puis plus d'amélioration
poly_sigma) flou avant 1.5 puis plus d'amélioration
"""


#Ouverture du flux video
cap = cv2.VideoCapture("../videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")

ret, frame1 = cap.read()  # Passe à l'image suivante
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Passage en niveaux de gris
hsv = np.zeros_like(frame1)  # Image nulle de même taille que frame1 (affichage OF)
hsv[:, :, 1] = 255  # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

while (ret):
    index += 1
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None,
                                        pyr_scale=0.5,  # Taux de réduction pyramidal
                                        levels=3,  # Nombre de niveaux de la pyramide
                                        winsize=15,
                                        # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations=3,  # Nb d'itérations par niveau
                                        poly_n=7,  # Taille voisinage pour approximation polynomiale
                                        poly_sigma=1.5,  # E-T Gaussienne pour calcul dérivé
                                        flags=0)
    histr = cv2.calcHist([flow[:, :, 0], flow[:, :, 1]], [0, 1], None, [512, 512], [-1, 1, -1, 1])

    histr = numpy.log(histr)/numpy.log(histr.max())*255
    histr[histr == -np.inf] = 0
    histr = histr.astype(np.uint8)

    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])  # Conversion cartésien vers polaire
    hsv[:, :, 0] = (ang * 180) / (2 * np.pi)  # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:, :, 2] = (mag * 255) / np.amax(mag)  # Valeur <--> Norme

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2, bgr))
    cv2.imshow('Image et Champ de vitesses (Farneback)', result)

    histr = cv2.applyColorMap(histr, cv2.COLORMAP_JET)

    cv2.imshow('Histo', histr)

    k = cv2.waitKey(15) & 0xff
    if k == 27:  # escape
        break
    elif k == ord('s'):  # or index == 100:
        cv2.imwrite('Frame_%04d_%d.png' % (index, time.time()), frame2)
        cv2.imwrite('OF_hsv_%04d_%d.png' % (index, time.time()), bgr)

    prvs = next
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

cap.release()
cv2.destroyAllWindows()
