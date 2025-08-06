import os
import logging
import random
from tabnanny import verbose

import numpy as np
import tensorflow as tf
from fontTools.varLib.interpolatable import test_gen
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

# Call backs
def get_callbacks(checkpoint_path):
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    def scheduler(epoch, lr):
        if epoch < 10:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))
    lr_callback = LearningRateScheduler(scheduler)
    return [checkpoint, early_stopping, reduce_lr, lr_callback]

# Compute class weights
def get_class_weights(generator):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(generator.classes), y=generator.classes)
    return {i: class_weights[i] for i in range(len(class_weights))}

# Training function
def train_model():
    set_seeds()
    logging.info(f"Using data from : {BASE_DIR}")
    train_gen, val_gen, test_gen = get_data_generators()
    model = build_model()
    if os.path.exists(CHECKPOINT_PATH):
        logging.info("Loading weights from checkpoint...")
        model.load_weights(CHECKPOINT_PATH)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        class_weights = get_class_weights(train_gen)
        callbacks = get_callbacks(CHECKPOINT_PATH)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=27,
            class_weight=class_weights,
            callbacks=callbacks
        )
        return model, history, val_gen, test_gen

# Evaluation function
def evaluate_model(model, val_gen, test_gen, history):
    best_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    val_loss, val_acc = best_model.evaluate(val_gen, verbose=0)
    test_loss, test_acc = best_model.evaluate(test_gen, verbose=0)
    logging.info(f"Validation loss: {val_loss}, Validation accuracy: {val_acc * 100:.2f}%")
    logging.info(f"Test loss: {test_loss}, Test Accuracy: {test_acc * 100:.2f}%")

    # Predictions
    test_steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
    y_pred = best_model.predict(test_gen, steps=test_steps, verbose=1)
    y_pred_labels = (y_pred > 0.5).astype(int)
    y_true = test_gen.classes[:len(y_pred_labels)]

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_labels)
    class_names = CLASSES
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

    # Classification report
    report = classification_report(y_true, y_pred_labels, target_names=class_names)
    print(report)

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,  'accuracy.png'))
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss.png'))
    plt.close()


if __name__ == "__main__":
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model, history, val_gen = train_model() # Train the model
    evaluate_model(model, val_gen, test_gen, history) # Evaluate and plot results






