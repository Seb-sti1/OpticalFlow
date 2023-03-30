import os
import sys
import time

import cv2
import joblib
import numpy as np
import pandas as pd

import Metrics

np.seterr(divide='ignore')
DISPLAY_HISTOGRAMS = int(sys.argv[2])
DISPLAY_FLOW = int(sys.argv[3])
DISPLAY_MAIN_FRAME = int(sys.argv[4])

# Path pour les frames descriptives de chaque plan
folder_path = "../plans"

if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# Séparation de plans
nb_hists = 5
hist_idx = 0  # Keep track of idx of hist to add the last one in the array of hists with minimal computation
threshold_gain = 3.5
hists_array = np.zeros(shape=(nb_hists, 256, 256, 3))
SAD = 0
hist_idx = 0
plan_idx = 0
# Paramètres du cercle indicateur de changement de plan
center = (50, 50)  # Coordonnées (x, y) du centre du cercle
radius = 25  # Rayon du cercle
red = (0, 0, 255)  # Couleur du cercle en format BGR (rouge)
thickness = -1  # Épaisseur du cercle (-1 pour remplir le cercle)

# Chargement du k-NN entrainé
model_path = '../models/knn_plan.joblib'
knn = joblib.load(model_path)
type_plans = ['plan fixe', 'pano horizontal', 'pano vertical', 'rotation', 'trav horizontal', 'trav vertical',
              'trav avant', 'trav arriere', 'zoom avant', 'zoom arriere']
X_columns = ["Vx_entropy", "Vy_entropy", "Vx_amplitude", "Vy_amplitude", "Vx_max", "Vy_max", "Vx_min", "Vy_min",
             "Vx_mean", "Vy_mean"]

# Paramètres pour affichage du type de plan
font = cv2.FONT_HERSHEY_SIMPLEX  # Police du texte
font_scale = 1  # Échelle de la police (taille du texte)
color = (255, 255, 255)  # Couleur du texte (B, G, R)
thickness = 2  # Épaisseur des lignes du texte

# Ouverture du flux video
video_path = "../videos/"
cap = cv2.VideoCapture(video_path + sys.argv[1])

ret, frame1 = cap.read()  # Passe à l'image suivante

if frame1 is None:
    print("Erreur")

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Passage en niveaux de gris
hsv = np.zeros_like(frame1)  # Image nulle de même taille que frame1 (affichage OF)
hsv[:, :, 1] = 255  # Toutes les couleurs sont saturées au maximum
index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Identification de la frame principale
main_frame_min = frame1
main_frame_min_entropy = (np.inf, np.inf)  # (Vx_entropy, Vy_entropy)
main_frame_max = frame1
main_frame_max_entropy = (-np.inf, -np.inf)

# Votes pour identification du type de plan
votes_plan = np.zeros(9)

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

    # Identification du type de plan
    histr = cv2.calcHist([flow[:, :, 0], flow[:, :, 1]], [1, 0], None, [512, 512], [-1, 1, -1, 1])

    X = Metrics.get_X_vector(flow, histr)  # Calcul à partir de l'histogramme et du flow par facilité des formules
    currentEntropy = (X[0], X[1])
    X = pd.DataFrame(data=[X], columns=X_columns)
    type_plan = knn.predict(X)[0]
    votes_plan[type_plan] += 1

    histr = np.log(histr) / np.log(histr.max()) * 255
    histr[histr == -np.inf] = 0
    histr = histr.astype(np.uint8)

    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])  # Conversion cartésien vers polaire
    hsv[:, :, 0] = (ang * 180) / (2 * np.pi)  # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:, :, 2] = (mag * 255) / np.amax(mag)  # Valeur <--> Norme
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    histr = cv2.applyColorMap(histr, cv2.COLORMAP_JET)

    # Identification de la frame principale
    if currentEntropy > main_frame_max_entropy:
        main_frame_max_entropy = currentEntropy
        main_frame_max = frame2
    elif currentEntropy < main_frame_min_entropy:
        main_frame_min_entropy = currentEntropy
        main_frame_min = frame2

    #  Detection de changement de plan
    frameYuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
    hist = cv2.calcHist([frameYuv], [1, 2], None, [256, 256], [0, 255, 0, 255])
    hist = np.log(hist)
    hist = hist / hist.max() * 255
    hist = hist.astype(np.uint8)
    hist = cv2.applyColorMap(hist, cv2.COLORMAP_JET)

    # Sum of Absolute Differencies of the current histogram
    SAD = np.sum(np.abs(hist - hists_array[hist_idx - 1]))
    # init threshold to 0
    threshold = 0

    # Sum of Absolute Differencies to compare the current and previous histograms
    for i in range(1, nb_hists):
        threshold += np.sum(np.abs(hists_array[i] - hists_array[i - 1]))
    # Adjustment of the threshold with threshold_gain
    threshold = threshold_gain * threshold / nb_hists

    if SAD > threshold:
        print('\a')  # Fait un bip sonore
        print("Changement de plan       SAD : ", SAD, "      threshold", threshold, "        SAD/threshold",
              SAD / threshold)
        # Reset les paramètres de détection
        main_frame_min_entropy = (np.inf, np.inf)  # (Vx_entropy, Vy_entropy)
        main_frame_max_entropy = (-np.inf, -np.inf)
        votes_plan = np.zeros(9)
        print(type_plans[np.argmax(votes_plan)])
        frame_to_save = cv2.putText(main_frame_max, f"Plan {plan_idx} : {type_plans[np.argmax(votes_plan)]}",
                                    org=(50, 40), fontFace=font, fontScale=font_scale, color=color, thickness=thickness)
        cv2.imwrite(f"{folder_path}/{sys.argv[1]}_plan_{plan_idx}_{type_plans[np.argmax(votes_plan)]}.png", frame_to_save)
        frame2 = cv2.circle(frame2, center, radius, red, -1)
        plan_idx += 1

    # Add the current histogram to the list for the upcoming histogram of the next iteration
    hists_array[hist_idx] = hist
    # Update the hist_idx to access the latest histogram
    hist_idx = hist_idx + 1 if (hist_idx < nb_hists - 1) else 0

    # Affichage des histogrammes et frames
    if DISPLAY_FLOW:
        result = np.vstack((frame2, bgr))
        cv2.imshow('Image et Champ de vitesses (Farneback)', result)
    else:
        cv2.imshow('Extrait', frame2)
    if DISPLAY_HISTOGRAMS:
        cv2.imshow('Histogramme vitesses', histr)
        cv2.imshow('Histogramme uv', hist)
    if DISPLAY_MAIN_FRAME:
        cv2.imshow('Frames principales (max en haut, min en bas)', np.vstack((main_frame_max, main_frame_min)))

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
