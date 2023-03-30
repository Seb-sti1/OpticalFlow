import cv2
import numpy as np

# Ouverture du flux video
cap = cv2.VideoCapture("../videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
# cap = cv2.VideoCapture("../videos/Rotation_OX(Tilt).m4v")
# cap = cv2.VideoCapture("../videos/Rotation_OY(Pan).m4v")
# cap = cv2.VideoCapture("../videos/Rotation_OZ(Roll).m4v")
# cap = cv2.VideoCapture("../videos/ZOOM_O_TRAVELLING.m4v")
# cap = cv2.VideoCapture("../videos/Travelling_OX.m4v")
# cap = cv2.VideoCapture("../videos/Travelling_OZ.m4v")
# cap = cv2.VideoCapture("../videos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
# cap = cv2.VideoCapture('../videos/Extrait1-Cosmos_Laundromat1(340p).m4v')

# Convention opencv
# Si entier alors entre 0 et 255
# Si float entre 0 et 1

# Paramètres du cercle indicateur de changement de plan
center = (50, 50)  # Coordonnées (x, y) du centre du cercle
radius = 25  # Rayon du cercle
color = (0, 0, 255)  # Couleur du cercle en format BGR (rouge)
thickness = -1  # Épaisseur du cercle (-1 pour remplir le cercle)

# Check if the video file was successfully loaded
if not cap.isOpened():
    print("Error loading video file")

nb_hists = 5
hist_idx = 0  # Keep track of idx of hist to add the last one in the array of hists with minimal computation
threshold_gain = 3.5
hists_array = np.zeros(shape=(nb_hists, 256, 256, 3))
SAD = 0

hist_idx = 0
# Loop through the video frames
while cap.isOpened():
    # Read the next frame from the video file
    ret, frame = cap.read()

    frameYuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # Don't forget the [] around the frame
    hist = cv2.calcHist([frameYuv], [1, 2], None, [256, 256], [0, 255, 0, 255])
    # Log
    hist = np.log(hist)
    # Normalization
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
        frame = cv2.circle(frame, center, radius, color, thickness)
        print("Changement de plan       SAD : ", SAD, "      threshold", threshold, "        SAD/threshold",
              SAD / threshold)

    # Add the current histogram to the list for the upcoming histogram of the next iteration
    hists_array[hist_idx] = hist
    # Update the hist_idx to access the latest histogram
    hist_idx = hist_idx + 1 if (hist_idx < nb_hists - 1) else 0

    # Check if there are no more frames
    if not ret:
        break

    # Display the frame
    cv2.imshow('frame', frame)
    cv2.imshow('histogramme', hist)

    # Wait for a key press
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()
