import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# Ayarlar
# =========================
IMG_SIZE = 145
SEED = 42
TEST_SIZE = 0.30
BATCH_SIZE = 32
EPOCHS = 25

CLASS_NAMES = ["yawn", "no_yawn", "Closed", "Open"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Kendi bilgisayarına göre güncelle
DATASET_DIR = Path(r"C:\Users\pv\drowsiness_dataset\train")

# =========================
# Yardımcı fonksiyonlar
# =========================
def read_and_resize_image(image_path: Path, img_size: int = IMG_SIZE) -> np.ndarray | None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Uyarı: Görsel okunamadı -> {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img


def load_yawn_data(dataset_dir: Path) -> list[tuple[np.ndarray, int]]:
    data = []
    for class_name in ["yawn", "no_yawn"]:
        class_dir = dataset_dir / class_name
        class_index = CLASS_TO_INDEX[class_name]

        if not class_dir.exists():
            print(f"Uyarı: Klasör bulunamadı -> {class_dir}")
            continue

        for image_name in os.listdir(class_dir):
            image_path = class_dir / image_name
            img = read_and_resize_image(image_path)

            if img is not None:
                data.append((img, class_index))
    return data


def load_eye_state_data(dataset_dir: Path) -> list[tuple[np.ndarray, int]]:
    data = []
    for class_name in ["Closed", "Open"]:
        class_dir = dataset_dir / class_name
        class_index = CLASS_TO_INDEX[class_name]

        if not class_dir.exists():
            print(f"Uyarı: Klasör bulunamadı -> {class_dir}")
            continue

        for image_name in os.listdir(class_dir):
            image_path = class_dir / image_name
            img = read_and_resize_image(image_path)

            if img is not None:
                data.append((img, class_index))
    return data


def build_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    yawn_data = load_yawn_data(dataset_dir)
    eye_data = load_eye_state_data(dataset_dir)

    all_data = yawn_data + eye_data
    if not all_data:
        raise ValueError("Hiç veri yüklenemedi. Dataset yollarını kontrol et.")

    X = np.array([item[0] for item in all_data], dtype=np.float32) / 255.0
    y = np.array([item[1] for item in all_data], dtype=np.int32)

    return X, y


def build_model(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_history(history: tf.keras.callbacks.History) -> None:
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss")
    plt.show()


def prepare_single_image(image_path: Path, img_size: int = IMG_SIZE) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Görsel okunamadı: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(model: tf.keras.Model, image_path: Path) -> str:
    sample = prepare_single_image(image_path)
    probs = model.predict(sample, verbose=0)
    pred_idx = int(np.argmax(probs, axis=1)[0])
    return CLASS_NAMES[pred_idx]


# =========================
# Ana akış
# =========================
def main():
    X, y = build_dataset(DATASET_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(CLASS_NAMES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate and print accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Generate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot()
    plt.show()

    model.save("drowsiness_model_mlp.keras")
    print("\nModel kaydedildi: drowsiness_model_mlp.keras")

    # Örnek tahmin
    sample_image = DATASET_DIR / "closed" / "3.jpg"
    if sample_image.exists():
        result = predict_image(model, sample_image)
        print(f"\nÖrnek tahmin: {sample_image.name} -> {result}")


if __name__ == "__main__":
    main()
