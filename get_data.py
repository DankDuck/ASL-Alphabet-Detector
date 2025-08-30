import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import kagglehub
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

drawing_utils = mp.solutions.drawing_utils

path = kagglehub.dataset_download("grassknoted/asl-alphabet")
train_dir = os.path.join(path, "asl_alphabet_train/asl_alphabet_train")

classes = sorted(os.listdir(train_dir))
class_to_idx = {cls: i for i, cls in enumerate(classes)}

filepaths = []
labels = []
for cls in classes:
    cls_dir = os.path.join(train_dir, cls)
    for fname in os.listdir(cls_dir):
        filepaths.append(os.path.join(cls_dir, fname))
        labels.append(class_to_idx[cls])

print(f"Total images: {len(filepaths)}")

def process_image(filepath, label):
    image = cv.imread(filepath)
    if image is None:
        return None, None

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # MediaPipe expects RGB

    results = hands.process(image)
    hands.close()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = []
            for point in hand_landmarks.landmark:
                lm.extend([point.x, point.y, point.z])
            return lm, label
    return None, None

landmarks = []
labels_out = []

with ThreadPoolExecutor(max_workers=5) as executor:  # adjust for your CPU
    futures = [executor.submit(process_image, fp, lbl) for fp, lbl in zip(filepaths, labels)]

    for i, f in enumerate(tqdm(as_completed(futures), total=len(futures))):
        lm, lbl = f.result()
        if lm is not None:
            landmarks.append(lm)
            labels_out.append(lbl)

        # Save progress every 1000
        if i % 1000 == 0 and i > 0:
            np.save("landmarks_partial.npy", np.array(landmarks))
            np.save("labels_partial.npy", np.array(labels_out))
            print(f"✅ Saved checkpoint at {i} images")

np.save("train_landmarks.npy", np.array(landmarks))
np.save("train_labels.npy", np.array(labels_out))

print("🎉 Finished landmark extraction and saved all data.")
