import cv2
from deepface import DeepFace
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # read frame from webcam
    # run deep face function for analyze
    result = DeepFace.analyze(img_path=frame, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)

    emotion = result["dominant_emotion"]
    age = result["age"]
    gender = result["gender"]
    race = result["dominant_race"]
    
    emo = str(emotion)
    age = str(age)
    gender = str(gender)
    race = str(race)

    # read image
    if emo == 'angry':  
        logo = cv2.imread('Images\\angry.jfif')
    elif emo == 'disgust':  
        logo = cv2.imread('Images\\disgust.jfif')
    elif emo == 'fear':  
        logo = cv2.imread('Images\\fear.jfif')
    elif emo == 'happy':
        logo = cv2.imread('Images\\happy.jfif')
    elif emo == 'neutral':
        logo = cv2.imread('Images\\neutral.jfif')
    elif emo == 'sad':
        logo = cv2.imread('Images\\sad.jfif')
    elif emo == 'suprise':
        logo = cv2.imread('Images\\suprise.jfif')

    # resize image
    size = 100
    logo = cv2.resize(logo, (size, size))

    # Create a mask of logo
    img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    # Region of Image (ROI), where we want to insert logo
    roi = frame[-size-10:-10, -size-10:-10]
 
    # Set an index of where the mask is
    roi[np.where(mask)] = 0
    roi += logo
 
    cv2.imshow('WebCam', frame)

    if cv2.waitKey(1) & 0xff == ord('q'): # enter q for exit
        break

cap.release()
cv2.destroyAllWindows()