"""
train_model.py – Phase 1 only (no fine-tuning)
Best approach for CPU training — Phase 2 requires GPU to be effective.
"""

import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="dataset/PlantVillage")
parser.add_argument("--epochs",   default=15, type=int)
parser.add_argument("--batch",    default=8,  type=int)
parser.add_argument("--img_size", default=160, type=int)
args = parser.parse_args()

IMG_SIZE  = (args.img_size, args.img_size)
BATCH     = args.batch
EPOCHS    = args.epochs
MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "plant_disease_model.keras")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

print("📂 Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    args.data_dir, validation_split=0.2, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.utils.image_dataset_from_directory(
    args.data_dir, validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH)

NUM_CLASSES = len(train_ds.class_names)
print(f"✅ Found {NUM_CLASSES} classes")

with open(os.path.join(os.path.dirname(MODEL_OUT), "class_names.json"), "w") as f:
    json.dump(train_ds.class_names, f)
print("✅ Class names saved")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(500).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

print("🏗️  Building model...")
base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False  # frozen — Phase 1 only

inputs  = layers.Input(shape=(*IMG_SIZE, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.RandomFlip("horizontal_and_vertical")(x)
x = layers.RandomRotation(0.2)(x)
x = layers.RandomZoom(0.1)(x)
x = layers.RandomBrightness(0.1)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

cb = [
    callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1),
]

print("\n🚀 Training (Phase 1 only — stable for CPU)...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

print(f"\n✅ Model saved → {MODEL_OUT}")