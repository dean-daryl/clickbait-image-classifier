import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/clickbait_mobilenet.h5")
TRAIN_DIR = os.path.join(os.path.dirname(__file__), "../data/train")
TEST_DIR = os.path.join(os.path.dirname(__file__), "../data/test")

def get_data_generators(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
):
    """
    Create ImageDataGenerators for training and testing datasets with enhanced augmentation.
    """
    # Data generators with augmentation for training, only rescaling for test
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=seed
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        seed=seed
    )

    print('Class indices:', train_gen.class_indices)
    return train_gen, test_gen

def build_mobilenet_model(input_shape=(128, 128, 3)):
    """
    Build a MobileNetV2-based model for binary image classification.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
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
    Train the MobileNetV2 model with transfer learning and save the best model to disk.
    """
    # Get data generators
    train_gen, test_gen = get_data_generators(train_dir, test_dir, IMAGE_SIZE, batch_size)

    # Build model
    model = build_mobilenet_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Call the model once with dummy data to build it
    dummy_input = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
    _ = model.predict(dummy_input)
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

    # Calculate class weights for balanced training
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )

    print(f"Class weights: {dict(enumerate(class_weights))}")

    # Train the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        callbacks=[checkpoint, early_stop],
        class_weight=dict(enumerate(class_weights))
    )

    print(f"Training complete. Best model saved to {model_path}")
    return model, history

def fine_tune_model(
    model_path=MODEL_PATH,
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    epochs=10,
    batch_size=BATCH_SIZE,
    fine_tune_at=100
):
    """
    Fine-tune the pre-trained model by unfreezing some layers.
    """
    # Load the trained model
    model = load_model(model_path)

    # Get data generators
    train_gen, test_gen = get_data_generators(train_dir, test_dir, IMAGE_SIZE, batch_size)

    # Unfreeze the base model
    base_model = model.layers[0]
    base_model.trainable = True

    # Fine-tune from this layer onwards
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001/10),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Update model path for fine-tuned version
    fine_tuned_path = model_path.replace('.h5', '_fine_tuned.h5')

    # Callbacks
    checkpoint = ModelCheckpoint(fine_tuned_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )

    # Fine-tune the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        callbacks=[checkpoint, early_stop],
        class_weight=dict(enumerate(class_weights))
    )

    print(f"Fine-tuning complete. Model saved to {fine_tuned_path}")
    return model, history

def evaluate_model(model_path=MODEL_PATH, test_dir=TEST_DIR, batch_size=BATCH_SIZE):
    """
    Evaluate the trained model on the test set.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    _, test_gen = get_data_generators(TRAIN_DIR, test_dir, IMAGE_SIZE, batch_size)
    model = load_model(model_path)

    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    return accuracy, y_pred, y_true

def preprocess_single_image(image_path, image_size=IMAGE_SIZE):
    """
    Load and preprocess a single image for prediction.
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_single_image(image_path, model_path=MODEL_PATH):
    """
    Predict if a single image is clickbait or not.
    """
    model = load_model(model_path)
    img_array = preprocess_single_image(image_path)
    prediction = model.predict(img_array)[0][0]

    class_label = "clickbait" if prediction > 0.5 else "not_clickbait"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return {
        'prediction': prediction,
        'class': class_label,
        'confidence': confidence
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate the clickbait image classifier.")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    args = parser.parse_args()

    if args.train:
        print("Starting training with MobileNetV2...")
        train_model(epochs=args.epochs, batch_size=args.batch_size)

    if args.fine_tune:
        print("Starting fine-tuning...")
        fine_tune_model(epochs=args.epochs, batch_size=args.batch_size)

    if args.evaluate:
        print("Evaluating model...")
        evaluate_model(batch_size=args.batch_size)

    if args.predict:
        print(f"Predicting image: {args.predict}")
        result = predict_single_image(args.predict)
        print(f"Prediction: {result['class']} (confidence: {result['confidence']:.4f})")
