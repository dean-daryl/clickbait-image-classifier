import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from preprocessing import get_data_generators, IMAGE_SIZE, BATCH_SIZE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/clickbait_cnn.h5")
TRAIN_DIR = os.path.join(os.path.dirname(__file__), "../data/train")
TEST_DIR = os.path.join(os.path.dirname(__file__), "../data/test")

def build_cnn_model(input_shape=(128, 128, 3)):
    """
    Build a simple CNN model for binary image classification.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    model_path=MODEL_PATH,
    epochs=20,
    batch_size=BATCH_SIZE
):
    """
    Train the CNN model and save the best model to disk.
    """
    train_gen, val_gen = get_data_generators(train_dir, test_dir, image_size=IMAGE_SIZE, batch_size=batch_size)

    model = build_cnn_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    print(f"Training complete. Best model saved to {model_path}")
    return model, history

def evaluate_model(model_path=MODEL_PATH, test_dir=TEST_DIR, batch_size=BATCH_SIZE):
    """
    Evaluate the trained model on the test set.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    _, test_gen = get_data_generators(TRAIN_DIR, test_dir, image_size=IMAGE_SIZE, batch_size=batch_size)
    model = load_model(model_path)

    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate the clickbait image classifier.")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    args = parser.parse_args()

    if args.train:
        train_model(epochs=args.epochs)
    if args.evaluate:
        evaluate_model()
