import cv2
import numpy as np
import tensorflow as tf
import time
import random
from helpers import get_top3, translate_to_ukrainian, put_unicode_text

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe import Image, ImageFormat

model = tf.keras.models.load_model("../model/asl_cnn_3.keras")

CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9',
               'a','b','c','d','e','f','g','h','i','j',
               'k','l','m','n','o','p','q','r','s','t',
               'u','v','w','x','y','z']

TEST_LETTERS = [c for c in CLASS_NAMES if c.isalpha()]
TIME_PER_LETTER = 5

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../model/hand_landmarker.task"),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.8
)
landmarker = HandLandmarker.create_from_options(options)


# ── 1. Camera + model prediction ─────────────────────────────────────────────

def get_frame_prediction(cap, last_prediction_time, prediction_buffer, window_start):
    ret, frame = cap.read()
    if not ret:
        return None, "", [], False, last_prediction_time, prediction_buffer, window_start

    frame = cv2.flip(frame, 1)
    prediction = ""
    top3 = []
    new_prediction_ready = False
    now = time.time()

    # Convert frame for mediapipe and run hand detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.hand_landmarks:
            x_list = [int(lm.x * w) for lm in hand_landmarks]
            y_list = [int(lm.y * h) for lm in hand_landmarks]

            # Draw colored skeleton on black canvas to match training data format
            black_canvas = np.zeros_like(frame)
            points = [(x, y) for x, y in zip(x_list, y_list)]

            C_GREY   = (100, 100, 100)
            C_RED    = (60,  60,  215)
            C_BEIGE  = (175, 215, 240)
            C_BLUE   = (185, 110, 60)
            C_GREEN  = (115, 215, 115)
            C_YELLOW = (85,  205, 235)
            C_PURPLE = (130, 90,  115)

            lines = {
                C_GREY:   [(0,1),(0,5),(0,17),(5,9),(9,13),(13,17)],
                C_BEIGE:  [(1,2),(2,3),(3,4)],
                C_PURPLE: [(5,6),(6,7),(7,8)],
                C_YELLOW: [(9,10),(10,11),(11,12)],
                C_GREEN:  [(13,14),(14,15),(15,16)],
                C_BLUE:   [(17,18),(18,19),(19,20)]
            }
            for color, pairs in lines.items():
                for p1, p2 in pairs:
                    cv2.line(black_canvas, points[p1], points[p2], color, 2)

            dots = {
                C_RED:    [0,1,5,9,13,17],
                C_BEIGE:  [2,3,4],
                C_PURPLE: [6,7,8],
                C_YELLOW: [10,11,12],
                C_GREEN:  [14,15,16],
                C_BLUE:   [18,19,20]
            }
            for color, indices in dots.items():
                for i in indices:
                    cv2.circle(black_canvas, points[i], 3, color, -1)

            # Crop a square region around the hand to avoid distortion
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
            half_side = (max(xmax - xmin, ymax - ymin) + 40) // 2
            y1, y2 = max(0, cy - half_side), min(h, cy + half_side)
            x1, x2 = max(0, cx - half_side), min(w, cx + half_side)
            model_input_img = black_canvas[y1:y2, x1:x2]

            # Sample every 0.2s into the buffer
            if model_input_img.size != 0 and (now - last_prediction_time > 0.2):
                cv2.imshow("Model Input", model_input_img)
                rgb_crop = cv2.cvtColor(model_input_img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(rgb_crop, (160, 160))
                img_normalized = img_resized / 255.0
                img_expanded = np.expand_dims(img_normalized, axis=0)

                pred = model.predict(img_expanded, verbose=0)
                class_id = np.argmax(pred)
                raw_prediction = CLASS_NAMES[class_id]
                prediction_buffer.append((now, raw_prediction, pred))
                last_prediction_time = now

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 1.5s fallback — commit whatever was collected if hand stayed visible too long
        if now - window_start >= 1.5:
            if prediction_buffer:
                confident_samples = [(ts, lbl, p) for ts, lbl, p in prediction_buffer
                                     if float(p[0][np.argmax(p[0])]) >= 0.75]
                if confident_samples:
                    labels = [p[1] for p in confident_samples]
                    prediction = max(set(labels), key=labels.count)
                    winner_pred = next(p[2] for p in confident_samples if p[1] == prediction)
                    top3 = get_top3(winner_pred, CLASS_NAMES)
                    new_prediction_ready = True
                prediction_buffer.clear()
            window_start = now

    else:
        # Hand just left frame — commit immediately so the sign is clean
        if prediction_buffer:
            confident_samples = [(ts, lbl, p) for ts, lbl, p in prediction_buffer
                                 if float(p[0][np.argmax(p[0])]) >= 0.75]
            if confident_samples:
                labels = [p[1] for p in confident_samples]
                prediction = max(set(labels), key=labels.count)
                winner_pred = next(p[2] for p in confident_samples if p[1] == prediction)
                top3 = get_top3(winner_pred, CLASS_NAMES)
                new_prediction_ready = True
            prediction_buffer.clear()
        window_start = now  # fresh window ready for next sign

    return frame, prediction, top3, new_prediction_ready, last_prediction_time, prediction_buffer, window_start


# ── 2. Normal mode UI drawing ─────────────────────────────────────────────────

def draw_ui(frame, prediction, top3, subtitle_text, ukrainian_text):
    h_frame, w_frame = frame.shape[:2]

    cv2.putText(frame, f"Prediction: {prediction}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Top 3 candidates with confidence
    for i, (label, conf) in enumerate(top3):
        cv2.putText(frame, f"  {i+1}. {label}: {conf}%",
                    (10, 90 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Ukrainian translation bar
    cv2.rectangle(frame, (0, h_frame - 100), (w_frame, h_frame - 50), (20, 20, 20), -1)
    frame = put_unicode_text(frame, ukrainian_text,
                             (10, h_frame - 95), font_size=28, color=(100, 255, 100))

    # English subtitle bar
    cv2.rectangle(frame, (0, h_frame - 50), (w_frame, h_frame), (0, 0, 0), -1)
    cv2.putText(frame, subtitle_text,
                (10, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame


# ── 3. Normal mode loop ───────────────────────────────────────────────────────

def run():
    cap = cv2.VideoCapture(0)
    last_prediction_time = 0
    prediction = ""
    subtitle_text = ""
    MAX_SUBTITLE_LENGTH = 30
    ukrainian_text = ""
    top3 = []
    prediction_buffer = []
    window_start = time.time()

    while cap.isOpened():
        frame, new_prediction, new_top3, new_prediction_ready, last_prediction_time, prediction_buffer, window_start = \
            get_frame_prediction(cap, last_prediction_time, prediction_buffer, window_start)

        if frame is None:
            break

        if new_prediction_ready:
            prediction = new_prediction
            top3 = new_top3
            subtitle_text += prediction
            if len(subtitle_text) >= MAX_SUBTITLE_LENGTH:
                subtitle_text = ""

        frame = draw_ui(frame, prediction, top3, subtitle_text, ukrainian_text)
        cv2.imshow("ASL Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:        # Spacebar
            subtitle_text += " "
        elif key == 8:         # Backspace
            subtitle_text = subtitle_text[:-1]
        elif key == ord('c'):  # Clear all
            subtitle_text = ""
        elif key == ord('t'):  # Translate to Ukrainian
            ukrainian_text = translate_to_ukrainian(subtitle_text)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


# ── 4. Self-test mode loop ────────────────────────────────────────────────────

def run_test():
    cap = cv2.VideoCapture(0)
    last_prediction_time = 0
    prediction = ""
    top3 = []
    prediction_buffer = []
    window_start = time.time()

    score = 0
    total = 0
    target_letter = random.choice(TEST_LETTERS)
    letter_start_time = time.time()
    result_message = ""
    result_message_time = 0
    result_color = (255, 255, 255)

    while cap.isOpened():
        frame, new_prediction, new_top3, new_prediction_ready, last_prediction_time, prediction_buffer, window_start = \
            get_frame_prediction(cap, last_prediction_time, prediction_buffer, window_start)

        if frame is None:
            break

        if new_prediction_ready:
            prediction = new_prediction
            top3 = new_top3

            # Check if the signed letter matches the target
            if prediction == target_letter:
                score += 1
                total += 1
                result_message = "CORRECT! +1"
                result_color = (0, 255, 0)
                result_message_time = time.time()
                target_letter = random.choice(TEST_LETTERS)
                letter_start_time = time.time()

        elapsed = time.time() - letter_start_time
        time_left = max(0, TIME_PER_LETTER - elapsed)

        # Time ran out — count as wrong and move to next letter
        if time_left == 0:
            total += 1
            result_message = "TIME'S UP!"
            result_color = (0, 0, 255)
            result_message_time = time.time()
            target_letter = random.choice(TEST_LETTERS)
            letter_start_time = time.time()

        h_frame, w_frame = frame.shape[:2]

        # Dark panel at top for target letter display
        cv2.rectangle(frame, (0, 0), (w_frame, 170), (0, 0, 0), -1)

        letter_display = target_letter.upper()
        text_size = cv2.getTextSize(letter_display, cv2.FONT_HERSHEY_SIMPLEX, 4, 6)[0]
        text_x = (w_frame - text_size[0]) // 2
        cv2.putText(frame, "Sign this:", (text_x - 20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, letter_display, (text_x, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 6)

        cv2.putText(frame, f"Score: {score}/{total}",
                    (w_frame - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Countdown bar — green > orange > red as time runs out
        bar_width = int((time_left / TIME_PER_LETTER) * w_frame)
        bar_color = (0, 255, 0) if time_left > 2 else (0, 165, 255) if time_left > 1 else (0, 0, 255)
        cv2.rectangle(frame, (0, 170), (bar_width, 185), bar_color, -1)

        cv2.putText(frame, f"Prediction: {prediction}",
                    (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for i, (label, conf) in enumerate(top3):
            cv2.putText(frame, f"  {i+1}. {label}: {conf}%",
                        (10, 255 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        # Flash result message in the center for 1.5 seconds
        if result_message and (time.time() - result_message_time < 1.5):
            msg_size = cv2.getTextSize(result_message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            msg_x = (w_frame - msg_size[0]) // 2
            cv2.putText(frame, result_message, (msg_x, h_frame // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, result_color, 3)

        cv2.imshow("ASL Self-Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

