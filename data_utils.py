import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(data_dir, img_size=(224, 224), batch_size=16):
    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

if __name__ == "__main__":
    # Test generators
    train_gen, val_gen = get_data_generators('data')
    print(f"Classes: {train_gen.class_indices}")
    print(f"Number of training samples: {train_gen.samples}")
    print(f"Number of validation samples: {val_gen.samples}")
