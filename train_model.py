"""
train_model.py – Phase 1 + optional Phase 2 fine-tuning
Switches MobileNetV2 → EfficientNetV2S for significantly better accuracy.
Phase 2 works on CPU but is much faster on GPU.
"""

import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetV2S

# ── GPU memory growth ────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",     default="dataset/PlantVillage")
parser.add_argument("--epochs_p1",    default=15,   type=int,   help="Phase 1 epochs (frozen base)")
parser.add_argument("--epochs_p2",    default=10,   type=int,   help="Phase 2 epochs (fine-tune); 0 = skip")
parser.add_argument("--batch",        default=16,   type=int)
parser.add_argument("--img_size",     default=224,  type=int,   help="Recommended ≥224 for EfficientNet")
parser.add_argument("--unfreeze",     default=30,   type=int,   help="# of top base layers to unfreeze in Phase 2")
parser.add_argument("--label_smooth", default=0.1,  type=float, help="Label smoothing (0 = off)")
args = parser.parse_args()

IMG_SIZE  = (args.img_size, args.img_size)
BATCH     = args.batch
MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "plant_disease_model.keras")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
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
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# ── Augmentation (stronger than before) ──────────────────────────────────────
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="augmentation")

# ── Model ─────────────────────────────────────────────────────────────────────
# EfficientNetV2S includes its own preprocessing internally — no Rescaling needed.
print("🏗️  Building model (EfficientNetV2S)...")
base = EfficientNetV2S(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    include_preprocessing=True,   # handles normalization internally
)
base.trainable = False  # Phase 1: frozen

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = augment(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    # label_smoothing only works with CategoricalCrossentropy; handled below
)

# Use label smoothing via CategoricalCrossentropy + one-hot if requested
if args.label_smooth > 0:
    # Wrap dataset to convert sparse labels → one-hot
    def to_onehot(x, y):
        return x, tf.one_hot(y, NUM_CLASSES)
    train_ds_p1 = train_ds.map(to_onehot, num_parallel_calls=AUTOTUNE)
    val_ds_p1   = val_ds.map(to_onehot,   num_parallel_calls=AUTOTUNE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smooth)
    metric  = "categorical_accuracy"
else:
    train_ds_p1 = train_ds
    val_ds_p1   = val_ds
    loss_fn = "sparse_categorical_crossentropy"
    metric  = "accuracy"

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss=loss_fn,
    metrics=[metric],
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(tag="p1"):
    return [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.4, patience=2, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1),
        callbacks.CSVLogger(os.path.join(os.path.dirname(MODEL_OUT), f"history_{tag}.csv")),
    ]

# ── Phase 1 ───────────────────────────────────────────────────────────────────
print("\n🚀 Phase 1 — training head only (frozen EfficientNetV2S base)...")
model.fit(
    train_ds_p1, validation_data=val_ds_p1,
    epochs=args.epochs_p1, callbacks=make_callbacks("p1"),
)
print(f"✅ Phase 1 complete. Model saved → {MODEL_OUT}")

# ── Phase 2 (fine-tuning) ─────────────────────────────────────────────────────
if args.epochs_p2 > 0:
    print(f"\n🔧 Phase 2 — unfreezing top {args.unfreeze} layers for fine-tuning...")
    # Unfreeze the top N layers of the base
    base.trainable = True
    for layer in base.layers[:-args.unfreeze]:
        layer.trainable = False

    # Use sparse labels for phase 2 (simpler)
    model.compile(
        optimizer=optimizers.Adam(1e-5),   # much lower LR to avoid destroying pretrained weights
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"   Trainable params: {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs_p2, callbacks=make_callbacks("p2"),
    )
    print(f"✅ Phase 2 complete. Final model saved → {MODEL_OUT}")
else:
    print("\nℹ️  Phase 2 skipped (--epochs_p2 0). Add --epochs_p2 10 to enable fine-tuning.")