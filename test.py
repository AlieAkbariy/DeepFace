import math
from gaze_tracking import GazeTracking
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
gaze = GazeTracking()

def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]
    :return: Angles in degrees for each axis
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi


while cap.isOpened():
    success, image = cap.read()


    # Convert the color space from BGR to RGB and get Mediapipe results
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Convert the color space from RGB to BGR to display well with Opencv
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_coordination_in_real_world = np.array([
        [285, 528, 200],
        [285, 371, 152],
        [197, 574, 128],
        [173, 425, 108],
        [360, 574, 128],
        [391, 425, 108]
    ], dtype=np.float64)

    h, w, _ = image.shape
    face_coordination_in_image = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 9, 57, 130, 287, 359]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_coordination_in_image.append([x, y])

            face_coordination_in_image = np.array(face_coordination_in_image,
                                                  dtype=np.float64)

            # The camera matrix
            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                   [0, focal_length, h / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Use solvePnP function to get rotation vector
            success, rotation_vec, transition_vec = cv2.solvePnP(
                face_coordination_in_real_world, face_coordination_in_image,
                cam_matrix, dist_matrix)

            # Use Rodrigues function to convert rotation vector to matrix
            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

            result = rotation_matrix_to_angles(rotation_matrix)

            if  (result[0] < 30 and result[0]>-10) and (result[1]<30 and result[1]>-30) :
                txt="mid"
            elif (result[0]>25 and result[0]<35 ):
                txt="up lvl1"
            elif (result[0]>35 and result[0]<45 ):
                txt="up lvl2"
            elif (result[0]>45  ):
                txt="up lvl3"
            elif(result[0]<-10 and result[0]>-20):
                txt="down lvl1"
            elif(result[0]<-20 and result[0]>-30):
                txt="down lvl2"
            elif(result[0]<-30):
                txt="down lvl3"
            elif (result[1]>25 and result[1]<35 ):
                txt="left lvl1"
            elif (result[1]>35 and result[1]<45 ):
                txt="left lvl2"
            elif (result[1]>45  ):
                txt="left lvl3"
            elif(result[1]<-25 and result[1]>-35):
                txt="right lvl1"
            elif(result[1]<-35 and result[1]>-45):
                txt="right lvl2"
            elif(result[1]<-45):
                txt="right lvl3"
            cv2.putText(image, txt, (90, 30),cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            gaze.refresh(image)

            image = gaze.annotated_frame()
            text = ""
            text2=""
            if gaze.is_blinking():
                text = "Blinking"
            elif gaze.is_right():
                text = " right"
            elif gaze.is_left():
                text = " left"
            elif gaze.is_center():
                text = " center"
            if gaze.is_blinking():
                text2 = "Blinking"
            elif gaze.is_up():
                text2 = " up"
            elif gaze.is_up() is not True and gaze.is_close() is not True:
                text2 = "center"

            cv2.putText(image,text2 + text, (90, 95), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)


            """horizontal_ratio= gaze.horizontal_ratio()
            vertical_ratio= gaze.vertical_ratio()
            cv2.putText(frame, "vertical_ratio:  " + str(vertical_ratio), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "horizonal_ratio : " + str(horizontal_ratio), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            cv2.putText(image, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(image, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)"""
    #deepface 

    result = DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False)

    emotion = result["dominant_emotion"]
    #age = result["age"]
    #gender = result["gender"]
    #race = result["dominant_race"]
    
    emo = str(emotion)
    #age = str(age)
    #gender = str(gender)
    #race = str(race)
    #res =str(age +' ' + gender+ ' ' + race)

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
    roi = image[-size-10:-10, -size-10:-10]
 
    # Set an index of where the mask is
    roi[np.where(mask)] = 0
    roi += logo

    # show information of age, gender, race
    #cv2.putText(image, res, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
    
    cv2.imshow('Head Pose Angles', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()