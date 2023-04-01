import csv
import time
import cv2
import numpy as np
import Metrics

np.seterr(divide='ignore', invalid='ignore')
# Ouverture du flux video
# cap = cv2.VideoCapture("../videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
# cap = cv2.VideoCapture("../videos/Rotation_OX(Tilt).m4v")
# cap = cv2.VideoCapture("../videos/Rotation_OY(Pan).m4v")
cap = cv2.VideoCapture("../videos/Rotation_OZ(Roll).m4v")
# cap = cv2.VideoCapture("../videos/ZOOM_O_TRAVELLING.m4v")
# cap = cv2.VideoCapture("../videos/Travelling_OX.m4v")
# cap = cv2.VideoCapture("../videos/Travelling_OZ.m4v")
# cap = cv2.VideoCapture("../videos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
# cap = cv2.VideoCapture('../videos/Extrait1-Cosmos_Laundromat1(340p).m4v')
ret, frame1 = cap.read()  # Passe à l'image suivante

if frame1 is None:
    print("Erreur")

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Passage en niveaux de gris
hsv = np.zeros_like(frame1)  # Image nulle de même taille que frame1 (affichage OF)
hsv[:, :, 1] = 255  # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Ouverture du fichier csv et création du writer
csvfile = open('../data/data2.csv', 'a', newline='', encoding='utf-8')
writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

plan_dic = {"plan fixe": 0, "pano horizontal": 1, "pano vertical": 2, "rotation": 3, "trav horizontal": 4,
            "trav vertical": 5, "trav avant": 6, "trav arrière": 7, "zoom avant": 8, "zoom arrière": 9}
prompt = ""
for key in plan_dic.keys():
    prompt += f"{key}({str(plan_dic[key])})  "
prompt += " : "
print(prompt)

while (ret):
    index += 1

    # Renvoie numpy.ndarray a la même taille que les images d'entrée et contient les coordonnées du vecteur
    # de mouvement (u, v) pour chaque pixel. Les coordonnées du vecteur sont stockées sous forme de canaux,
    # où le premier canal (u) représente le déplacement horizontal et le deuxième canal (v) représente le déplacement vertical
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None,
                                        pyr_scale=0.5,  # Taux de réduction pyramidal
                                        levels=5,  # Nombre de niveaux de la pyramide
                                        winsize=15,
                                        # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations=3,  # Nb d'itérations par niveau
                                        poly_n=5,  # Taille voisinage pour approximation polynomiale
                                        poly_sigma=0.5,  # E-T Gaussienne pour calcul dérivé
                                        flags=0)

    # L'histogramme utilise -flow et [1, 0] au lieu de [0, 1] afin de faire correspondre visuellement les vitesses
    # (Points nombreux à gauche dans l'histogramme quand vitesse de l'image vers la gauche etc)
    histr = cv2.calcHist([flow[:, :, 0], flow[:, :, 1]], [1, 0], None, [512, 512], [-1, 1, -1, 1])

    X = Metrics.get_X_vector(flow, histr)  # Calcul à partir de l'histogramme et du flow par facilité des formules

    histr = np.log(histr) / np.log(histr.max()) * 255
    histr[histr == -np.inf] = 0
    histr = histr.astype(np.uint8)

    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])  # Conversion cartésien vers polaire
    hsv[:, :, 0] = (ang * 180) / (2 * np.pi)  # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:, :, 2] = (mag * 255) / np.amax(mag)  # Valeur <--> Norme

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2, bgr))

    histr = cv2.applyColorMap(histr, cv2.COLORMAP_JET)

    cv2.imshow('Histo', histr)
    cv2.imshow('Image et Champ de vitesses (Farneback)', result)

    k = cv2.waitKey(15) & 0xff
    if k in range(48, 58):
        svm_class = k - 48
        print(f"Classe {list(plan_dic.keys())[svm_class]}({svm_class})")
        print(prompt)
        if svm_class in list(plan_dic.values()):
            vector = X + [svm_class]
            writer.writerow(vector)
        else:
            print(svm_class, "not in available classes :", list(plan_dic.values()))
    elif k == 27:  # escape
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
# Fermer le fichier
csvfile.close()
