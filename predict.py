"""
predict.py – Command-line prediction for a single image
Usage:  python predict.py --image path/to/leaf.jpg
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", default=os.path.join(BASE, "model", "plant_disease_model_old.keras"))
args = parser.parse_args()

if not os.path.exists(args.model):
    sys.exit("❌ Model not found. Run train_model.py first.")

# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model(args.model)

# ✅ Read image size directly from the model (no hardcoding)
_, img_h, img_w, _ = model.input_shape
IMG_SIZE = (img_w, img_h)
print(f"ℹ️  Model expects input size: {IMG_SIZE}")

# ── Load class names from saved JSON (generated during training) ──────────────
class_names_path = os.path.join(BASE, "model", "class_names.json")
if os.path.exists(class_names_path):
    with open(class_names_path) as f:
        CLASS_NAMES = json.load(f)
    print(f"ℹ️  Loaded {len(CLASS_NAMES)} classes from class_names.json")
else:
    # Fallback hardcoded list
    CLASS_NAMES = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Apple___healthy", "Background_without_leaves", "Blueberry___healthy",
        "Cherry___Powdery_mildew", "Cherry___healthy",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust_",
        "Corn___Northern_Leaf_Blight", "Corn___healthy",
        "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
        "Peach___healthy", "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy", "Potato___Early_blight",
        "Potato___Late_blight", "Potato___healthy",
        "Raspberry___healthy", "Soybean___healthy",
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
        "Strawberry___healthy", "Tomato___Bacterial_spot",
        "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
    ]
    print("⚠️  class_names.json not found, using hardcoded fallback list.")

# ── Preprocess image ──────────────────────────────────────────────────────────
img = Image.open(args.image).convert("RGB").resize(IMG_SIZE)  # ✅ uses model's size
arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

# ── Predict ───────────────────────────────────────────────────────────────────
pred = model.predict(arr)[0]
top3 = np.argsort(pred)[::-1][:3]

print("\n🌿 PlantGuard Prediction")
print("─" * 40)
for rank, i in enumerate(top3, 1):
    label = CLASS_NAMES[i].replace("___", " → ").replace("_", " ")
    print(f"  #{rank}  {label:45s}  {pred[i]*100:.1f}%")

# ── Disease info lookup ───────────────────────────────────────────────────────
disease_csv = os.path.join(BASE, "data", "disease_info.csv")
if os.path.exists(disease_csv):
    df  = pd.read_csv(disease_csv, encoding="latin1")
    row = df.iloc[top3[0]]
    print("\n📋 Disease:", row["disease_name"])
    print("\n📖 Description:\n  ", str(row["description"])[:300], "…")
    print("\n💊 Treatment:\n  ", str(row["Possible Steps"])[:300], "…")
else:
    print(f"\n⚠️  disease_info.csv not found at {disease_csv} — skipping disease details.")

print()