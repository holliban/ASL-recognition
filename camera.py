import cv2
import numpy as np
import tensorflow as tf
import time

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
    min_hand_detection_confidence=0.8
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_prediction_time = 0
prediction = ""
subtitle_text = ""
MAX_SUBTITLE_LENGTH = 30  # characters before clear
new_prediction_ready = False

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

            # 1. Create a pure black canvas
            black_canvas = np.zeros_like(frame)
            points = [(x, y) for x, y in zip(x_list, y_list)]

            # Exact colors from the dataset (in BGR format for OpenCV)
            C_GREY = (100, 100, 100)
            C_RED = (60, 60, 215)
            C_BEIGE = (175, 215, 240)  # Thumb
            C_BLUE = (185, 110, 60)  # Index
            C_GREEN = (115, 215, 115)  # Middle
            C_YELLOW = (85, 205, 235)  # Ring
            C_PURPLE = (130, 90, 115)  # Pinky

            # 2. Draw lines
            lines = {
                C_GREY: [(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)],  # Palm base
                C_BEIGE: [(1, 2), (2, 3), (3, 4)],
                C_PURPLE: [(5, 6), (6, 7), (7, 8)],
                C_YELLOW: [(9, 10), (10, 11), (11, 12)],
                C_GREEN: [(13, 14), (14, 15), (15, 16)],
                C_BLUE: [(17, 18), (18, 19), (19, 20)]
            }
            for color, pairs in lines.items():
                for p1, p2 in pairs:
                    cv2.line(black_canvas, points[p1], points[p2], color, 2)

            # 3. Draw dots
            dots = {
                C_RED: [0, 1, 5, 9, 13, 17],  # Palm joints
                C_BEIGE: [2, 3, 4],
                C_PURPLE: [6, 7, 8],
                C_YELLOW: [10, 11, 12],
                C_GREEN: [14, 15, 16],
                C_BLUE: [18, 19, 20]
            }
            for color, indices in dots.items():
                for i in indices:
                    cv2.circle(black_canvas, points[i], 3, color, -1)

            # 4. Crop the black canvas around the hand
            # (a perfect square to prevent distortion)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
            half_side = (max(xmax - xmin, ymax - ymin) + 40) // 2
            y1, y2 = max(0, cy - half_side), min(h, cy + half_side)
            x1, x2 = max(0, cx - half_side), min(w, cx + half_side)
            model_input_img = black_canvas[y1:y2, x1:x2]

            # 5. Feed the cropped skeleton to the model
            if model_input_img.size != 0 and (time.time() - last_prediction_time > 1):
                cv2.imshow("Model Input", model_input_img)

                # Convert BGR to RGB
                rgb_crop = cv2.cvtColor(model_input_img, cv2.COLOR_BGR2RGB)

                img_resized = cv2.resize(rgb_crop, (160, 160))
                img_normalized = img_resized / 255.0
                img_expanded = np.expand_dims(img_normalized, axis=0)

                pred = model.predict(img_expanded, verbose=0)
                class_id = np.argmax(pred)
                prediction = CLASS_NAMES[class_id]
                new_prediction_ready = True
                last_prediction_time = time.time()

            # Draw the square box on your main video feed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Prediction: {prediction}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if new_prediction_ready:
        subtitle_text += prediction
        if len(subtitle_text) >= MAX_SUBTITLE_LENGTH:
            subtitle_text = ""
        new_prediction_ready = False

    # Subtitle bar at the bottom
    h_frame, w_frame = frame.shape[:2]
    cv2.rectangle(frame, (0, h_frame - 50), (w_frame, h_frame), (0, 0, 0), -1)
    cv2.putText(frame, subtitle_text,
                (10, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("ASL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == 32:  # Spacebar
        subtitle_text += " "
    elif key == 8:  # Backspace
        subtitle_text = subtitle_text[:-1]
    elif key == ord('c'):  # C to clear
        subtitle_text = ""

cap.release()
cv2.destroyAllWindows()
landmarker.close()
