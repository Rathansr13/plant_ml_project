"""
Plant Disease Detection Web Application
Flask backend - serves predictions and disease information
"""

import os
import io
import base64
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.keras")
DATA_DIR   = os.path.join(BASE_DIR, "data")
IMG_SIZE   = (160, 160)

# 39 classes matching the PlantVillage dataset order
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

# ── Load data ──────────────────────────────────────────────────────────────────
disease_df    = pd.read_csv(os.path.join(DATA_DIR, "disease_info.csv"),    encoding="latin1")
supplement_df = pd.read_csv(os.path.join(DATA_DIR, "supplement_info.csv"), encoding="latin1")

# ── Load model ────────────────────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print("⚠️  No saved model found. Train first using train_model.py")


# ── Helpers ──────────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def get_disease_info(class_name: str) -> dict:
    row = disease_df[disease_df["disease_name"].str.contains(
        class_name.split("___")[-1].replace("_", " "), case=False, na=False
    )]
    if row.empty:
        row = disease_df.iloc[int(CLASS_NAMES.index(class_name))]
    else:
        row = row.iloc[0]
    return {
        "disease_name":  row["disease_name"],
        "description":   row["description"],
        "possible_steps": row["Possible Steps"],
        "image_url":     row["image_url"],
    }

def get_supplement_info(class_name: str) -> dict:
    row = supplement_df[supplement_df["disease_name"] == class_name]
    if row.empty:
        return {}
    row = row.iloc[0]
    return {
        "supplement_name":  row.get("supplement name", ""),
        "supplement_image": row.get("supplement image", ""),
        "buy_link":         row.get("buy link", ""),
    }


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img   = Image.open(io.BytesIO(file.read()))
        arr   = preprocess_image(img)
        preds = model.predict(arr)[0]

        top3_idx  = np.argsort(preds)[::-1][:3]
        top3      = [{"class": CLASS_NAMES[i],
                      "confidence": float(round(preds[i] * 100, 2))}
                     for i in top3_idx]

        best_class = CLASS_NAMES[top3_idx[0]]
        disease    = get_disease_info(best_class)
        supplement = get_supplement_info(best_class)

        # Encode image for preview
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "top3":       top3,
            "disease":    disease,
            "supplement": supplement,
            "image_b64":  img_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/diseases")
def diseases():
    data = disease_df.to_dict(orient="records")
    return jsonify(data)


@app.route("/supplements")
def supplements():
    data = supplement_df.to_dict(orient="records")
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
