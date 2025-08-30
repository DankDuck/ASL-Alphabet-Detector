import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
import random
import threading
import os

from playsound3 import playsound 

IMAGE_SIZE = 300
OFFSET = 10

capture = cv.VideoCapture(0, cv.CAP_DSHOW)

model = tf.keras.models.load_model("asl_tracker_lm.keras")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
valid_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

def predict_letter(landmarks):
    if landmarks is not None:
        prediction = model.predict(landmarks)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        print("Confidence: " + str(confidence))

        return labels[predicted_class]
    else:
        print("No hand detected in image")

sound_lock = threading.Lock()

def play_sound(file):
    if sound_lock.locked():
        return
    
    def _play():
        with sound_lock:
            playsound(file)

    threading.Thread(target=_play, daemon=True).start()

def get_letter_image(letter):
    path = "png_letters/"
    for file_name in os.listdir(path):
        if letter == os.path.splitext(file_name)[0]: 
            return cv.imread(path + file_name)
        
def overlay_image(frame, overlay, x, y):
    h, w = overlay.shape[:2]
    frame[y: y+h, x: x+ w] = overlay 
    return frame
             
previous_frame_letter = "None"
consecutive_letters = 0

incorrect_frames = 0

frames_to_check = 100

letter_to_show = ""

random.shuffle(valid_labels)
letter_to_show = valid_labels[0]

total_correct = 0
total_wrong = 0
total = 0

while True:
    success, frame = capture.read()
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    cv.putText(frame, text="Show: " + letter_to_show, org=(300,100), fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 128, 50), thickness=4)

    if total:
        correct = total_correct - total_wrong
        cv.putText(frame, text=str(correct) + " / " + str(total), org=(50,50), fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 0, 255), thickness=3)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            predicted_letter = predict_letter(landmarks)

            print("Predicted letter: " + predicted_letter)
            if predicted_letter == previous_frame_letter:
                consecutive_letters += 1

                if consecutive_letters >= 10:
                    if predicted_letter == letter_to_show:
                        if incorrect_frames >= frames_to_check:
                            total_wrong += 1

                        valid_labels.pop(0)
                        total += 1
                        total_correct += 1
                        incorrect_frames = 0
                        consecutive_letters = 0
                        letter_to_show = valid_labels[0]
                        play_sound("ding.wav")
            else:
                consecutive_letters = 0
                previous_frame_letter = predicted_letter

    incorrect_frames += 1

    if incorrect_frames >= frames_to_check:
        letter_image = get_letter_image(letter_to_show)
        resized_image = cv.resize(letter_image, (150, 100))
        overlay_image(frame, resized_image, 50, 350)

    cv.imshow("ASL Tracker", frame)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break