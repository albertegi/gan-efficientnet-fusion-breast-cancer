import os
import logging
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seeds
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Get base directory from environment variable or defaults
BASE_DIR = os.environ.get("BREAST_CANCER_DATA", os.path.join(os.getcwd(), "data/input/BreakHis_v1/histology_slides/breast"))
TRAIN_DIR = os.path.join(BASE_DIR, 'train') # creates path to training data
VAL_DIR = os.path.join(BASE_DIR, 'validation')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'efficient_checkpoints')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.h5') # creates path to save the best model

# Discover class names dynamically
CLASSES = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]

# Utility: get image extensions
def get_image_extension(directory):
    exts = set()
    for cls in CLASSES:
        files = glob.glob(os.path.join(directory, cls, '*'))
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext: # if the file has an extension, add it to the exts set.
                exts.add(ext)
    return list(exts) # convert the set of extensions into to a list and return it.

IMAGE_EXTS = get_image_extension(TRAIN_DIR)

# Data generators

def get_data_generators(seed=42):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.9, 1.1],
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.5
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training',
        seed=seed
    )

    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training',
        seed=seed
    )

    test_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=seed
    )
    return train_generator, validation_generator, test_generator

# Model Definition
def build_model(dropout_rate=0.4):
    model = Sequential([
        EfficientNetB5(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
        GlobalAveragePooling2D,
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])
    return model






