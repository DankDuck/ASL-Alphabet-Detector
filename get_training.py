import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np 

X_data = np.load("train_landmarks.npy")
Y_data = np.load("train_labels.npy")

print("Data shape:", X_data.shape, "Labels shape:", Y_data.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_data, Y_data, test_size=0.2, random_state=42, stratify=Y_data
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(29, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test), 
    epochs=10
)

model.save("asl_tracker_lm.keras")