# Fundamental classes
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# Image related
import cv2
from PIL import Image

# For ploting
import matplotlib.pyplot as plt

# For the model and it's training
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Add, GlobalAveragePooling2D

# =========================
# SETTINGS
# =========================
TRAIN_DIR = r"C:\Users\pv\Desktop\GermanTrafficSignsDataset\Train"
classes = 43
IMG_SIZE = (30, 30)

data = []
labels = []

for i in range(classes):
    class_path = os.path.join(TRAIN_DIR, str(i))  # Train/0, Train/1, ..., Train/42

    if not os.path.isdir(class_path):
        print(f"[WARNING] Folder not found: {class_path}")
        continue

    images = os.listdir(class_path)
    for a in images:
        try:
            img_path = os.path.join(class_path, a)
            image = Image.open(img_path).convert("RGB")
            image = image.resize(IMG_SIZE)
            image = np.array(image, dtype=np.uint8)

            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error loading image: {img_path} -> {e}")

# Convert lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
print("Loaded:", data.shape, labels.shape)

# Normalization
data = data.astype("float32") / 255.0

# Split training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Displaying the shape after the split
print("Split shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# =========================
# RESNET ARCHITECTURE
# =========================

def residual_block(x, filters, stride=1, projection=False):
    """
    Residual block with optional projection shortcut.
    
    Args:
        x: Input tensor
        filters: Number of filters in conv layers
        stride: Stride for first conv layer (for downsampling)
        projection: Whether to use 1x1 conv on shortcut
    
    Returns:
        Output tensor after residual connection
    """
    shortcut = x
    
    # Main path
    x = Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut path
    if projection:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut to main path
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

# Build ResNet model using Functional API
inputs = Input(shape=(30, 30, 3))

# Initial convolution
x = Conv2D(64, (3, 3), strides=1, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Stage 1: 2 identity residual blocks with 64 filters (30x30x64)
x = residual_block(x, 64, stride=1, projection=False)
x = residual_block(x, 64, stride=1, projection=False)

# Stage 2: 1 projection block (stride=2) + 1 identity block with 128 filters (15x15x128)
x = residual_block(x, 128, stride=2, projection=True)
x = residual_block(x, 128, stride=1, projection=False)

# Stage 3: 1 projection block (stride=2) + 1 identity block with 256 filters (8x8x256)
x = residual_block(x, 256, stride=2, projection=True)
x = residual_block(x, 256, stride=1, projection=False)

# Global average pooling and classification
x = GlobalAveragePooling2D()(x)
outputs = Dense(43, activation='softmax')(x)

# Create Model
model = Model(inputs=inputs, outputs=outputs)

# =========================
# MODEL COMPILATION
# =========================
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display model architecture
model.summary()

# =========================
# MODEL TRAINING
# =========================
with tf.device('/GPU:0'):
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# =========================
# MODEL SAVING
# =========================
model.save('ResNet_GermanTrafficSign.keras')
print("Model saved to 'ResNet_GermanTrafficSign.keras'")

# =========================
# TRAINING VISUALIZATION
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training_Accuracy')
plt.plot(history.history['val_accuracy'], label='Val_Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training_Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# =========================
# PERFORMANCE METRICS
# =========================
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

y_true = np.argmax(y_test, axis=1)
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

Precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
Recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
F1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print(f"Precision: {Precision:.3f}")
print(f"Recall: {Recall:.3f}")
print(f"F1-Score: {F1:.3f}")

# =========================
# PERFORMANCE METRICS VISUALIZATION
# =========================
metrics_names = ["Precision", "Recall", "F1-Score"]
metrics_values = [Precision, Recall, F1]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_names, metrics_values)

plt.title("Performance Metrics (ResNet Model)")
plt.ylabel("Score")
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height, f"{height:.3f}",
        ha="center", va="bottom"
    )
plt.show()

# =========================
# CONFUSION MATRIX
# =========================
Confusion_Matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(16, 10))
display = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
display.plot(include_values=False, cmap="Blues", ax=plt.gca(), xticks_rotation="vertical")
plt.title("Confusion Matrix (GTSRB - 43 Classes)")
plt.show()
