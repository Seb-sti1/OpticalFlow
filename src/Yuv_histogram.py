import cv2
import numpy as np

cap = cv2.VideoCapture('videos/Extrait1-Cosmos_Laundromat1(340p).m4v')

# Convention opencv
# Si entier alors entre 0 et 255
# Si float entre 0 et 1

# Check if the video file was successfully loaded
if not cap.isOpened():
    print("Error loading video file")

# Loop through the video frames
while cap.isOpened():
    # Read the next frame from the video file
    ret, frame = cap.read()

    ranges = [0, 255]
    frameYuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Don't forget the [] around the frame
    hist = cv2.calcHist([frameYuv], [1,2], None, [256, 256], [0,255, 0, 255])
    hist = np.log(hist)
    hist = hist/ hist.max() * 255
    hist = hist.astype(np.uint8) # Normalization to
    hist = cv2.applyColorMap(hist, cv2.COLORMAP_JET)

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
