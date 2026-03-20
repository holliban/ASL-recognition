import cv2
import numpy as np
import tensorflow as tf
import time
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from mediapipe import Image, ImageFormat

model = tf.keras.models.load_model("model/asl_cnn_3.keras")

CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9',
               'a','b','c','d','e','f','g','h','i','j',
               'k','l','m','n','o','p','q','r','s','t',
               'u','v','w','x','y','z']

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="model/hand_landmarker.task"),
    running_mode=RunningMode.IMAGE,  # synchronous
    num_hands=1,
    min_hand_detection_confidence=0.7
)
landmarker = HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)
last_prediction_time = 0
prediction = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB numpy array (must be uint8, contiguous)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    # Wrap numpy array in mediapipe Image — correct way without mp.solutions
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.hand_landmarks:
            x_list = [int(lm.x * w) for lm in hand_landmarks]
            y_list = [int(lm.y * h) for lm in hand_landmarks]
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            padding = 20
            xmin, ymin = xmin - padding, ymin - padding
            xmax, ymax = xmax + padding, ymax + padding

            hand_img = frame[max(ymin,0):min(ymax,h), max(xmin,0):min(xmax,w)]
            if hand_img.size != 0 and (time.time() - last_prediction_time > 1):
                hand_img = cv2.resize(hand_img, (160, 160))
                hand_img = hand_img / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)
                pred = model.predict(hand_img, verbose=0)
                class_id = np.argmax(pred)
                prediction = CLASS_NAMES[class_id]
                last_prediction_time = time.time()

            # Draw bounding box
            cv2.rectangle(frame,
                          (max(xmin, 0), max(ymin, 0)),
                          (min(xmax, w), min(ymax, h)),
                          (0, 255, 0), 2)
            # Draw landmarks
            for lm in hand_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

    cv2.putText(frame, f"Prediction: {prediction}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()