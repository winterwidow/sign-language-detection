import cv2
import mediapipe as mp
import numpy as np
import joblib  # for loading the trained ML model

cap = cv2.VideoCapture(0)

model = joblib.load("model/asl_svm_model.pkl")  # trained model

# detect hands:
mp_hands = mp.solutions.hands
# load the hands module from mediapipe
# hands module is used to detect and track hands in images and videos

hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)
# allows tracking of hands in a video stream; only 1 hand detected; Sets the threshold for detection confidence to 0.7

mp_draw = mp.solutions.drawing_utils
# drawing_utils is used to draw the landmarks on the image - landmarks is the key points on the hand


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # ret is a boolean indicating if the frame was captured successfully
    # frame is the actual frame captured

    frame = cv2.flip(frame, 1)
    # flip the frame horizontally - mirror image

    x1, y1 = 100, 100
    x2, y2 = 400, 400
    roi = frame[y1:y2, x1:x2]
    # roi is the region of interest - the area where the hand is expected to be

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # convert the roi from BGR to RGB - mediapipe works with RGB images
    results = hands.process(roi_rgb)
    # process the rgb image to detect hands

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # draw a rectangle around the roi

    if results.multi_hand_landmarks:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                """# extract Landmark Features for Classification
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                """

                # --------- Predict Sign Using trained image-based SVM model ---------
                roi_resized = cv2.resize(roi, (64, 64))
                roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                roi_flatten = roi_gray.flatten().reshape(1, -1)
                prediction = model.predict(roi_flatten)[0]

                # display prediction

                cv2.putText(
                    frame,
                    f"Prediction: {prediction}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

    cv2.imshow(
        "Sign Language Input", frame
    )  # imshow is used to display an image in a window

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # & 0xFF is used to get the last 8 bits of the key code
    # basically is user presses 'q', the loop will break

cap.release()
cv2.destroyAllWindows()
