import os
from typing import Tuple, List
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img, save_img

# Configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

def get_data_generators(
    train_dir: str,
    test_dir: str,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED
):
    """
    Create ImageDataGenerators for training and testing datasets.

    Args:
        train_dir (str): Path to training data directory.
        test_dir (str): Path to testing data directory.
        image_size (Tuple[int, int]): Target size for images.
        batch_size (int): Batch size.
        seed (int): Random seed.

    Returns:
        train_generator, test_generator: Keras DirectoryIterator objects.
    """
    # Data augmentation for training
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

    # Only rescaling for test/validation
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=seed
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        seed=seed
    )

    print('Class indices:', train_generator.class_indices)
    return train_generator, test_generator

def preprocess_single_image(image_path: str, image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Load and preprocess a single image for prediction.

    Args:
        image_path (str): Path to the image file.
        image_size (Tuple[int, int]): Target size for the image.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if __name__ == "__main__":
    # Example usage for testing
    train_dir = os.path.join(os.path.dirname(__file__), "../data/train")
    test_dir = os.path.join(os.path.dirname(__file__), "../data/test")
    train_gen, test_gen = get_data_generators(train_dir, test_dir)
    print("Class indices:", train_gen.class_indices)
    print("Number of training samples:", train_gen.samples)
    print("Number of test samples:", test_gen.samples)
