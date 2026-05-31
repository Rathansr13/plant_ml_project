# 🌿 PlantGuard AI – Plant Disease Detection

An end-to-end deep learning web application that detects **39 plant diseases** across 14 crop types using a fine-tuned MobileNetV2 model.

---

## 📁 Project Structure

```
plant_disease_detection/
├── app.py                  # Flask web server
├── train_model.py          # Model training script
├── predict.py              # CLI inference tool
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container config
├── data/
│   ├── disease_info.csv    # 39 diseases: name, description, treatment
│   └── supplement_info.csv # Recommended supplements + buy links
├── model/
│   └── plant_disease_model.h5  # (generated after training)
├── templates/
│   └── index.html          # Frontend UI
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## 🚀 Step-by-Step Execution Guide

### STEP 1 – Clone / Set Up Project

```bash
# Create and enter project folder
mkdir plant_disease_detection && cd plant_disease_detection

# (Copy all project files here)
```

---

### STEP 2 – Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

---

### STEP 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow requires Python 3.9–3.12. GPU training is optional but speeds up training 10×.

---

### STEP 4 – Download the PlantVillage Dataset

The model trains on the **PlantVillage** dataset (~3.5 GB, 87,000 images).

```bash
# Option A – Kaggle CLI (recommended)
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d dataset/

# Option B – Manual download
# Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# Download & extract to: dataset/PlantVillage/
```

Expected folder structure after extraction:
```
dataset/PlantVillage/
├── Apple___Apple_scab/         (images)
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
├── ...  (39 class folders)
```

---

### STEP 5 – Train the Model

```bash
python train_model.py \
    --data_dir dataset/PlantVillage \
    --epochs 15 \
    --batch 32

# On CPU only (slower):
python train_model.py --data_dir dataset/PlantVillage --epochs 10 --batch 16
```

**What happens:**
- Phase 1 (epochs 1–7): Trains the classification head only (MobileNetV2 frozen)
- Phase 2 (epochs 8–15): Fine-tunes the top 40 backbone layers at a lower LR
- Best model auto-saved to `model/plant_disease_model.h5`

**Expected accuracy:** ~96–98% on validation set

**Training time:**
| Hardware    | Time      |
|-------------|-----------|
| GPU (NVIDIA)| ~25 min   |
| CPU only    | ~3–6 hours|

---

### STEP 6 – Test with a Single Image (Optional)

```bash
python predict.py --image path/to/your/leaf.jpg
```

Sample output:
```
🌿 PlantGuard Prediction
────────────────────────────────────────
  #1  Tomato → Early Blight              94.3%
  #2  Tomato → Target Spot               3.1%
  #3  Tomato → Septoria leaf spot        1.8%

📋 Disease: Tomato : Early Blight
📖 Description: Early blight is caused by Alternaria solani…
💊 Treatment: Remove infected leaves immediately…
```

---

### STEP 7 – Run the Web Application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

### STEP 8 – Using the Web App

1. **Upload** – Drag & drop or browse a leaf image
2. **Analyze** – Click "Analyze Plant"
3. **Results** – See:
   - Top-3 predictions with confidence %
   - Disease name, description, reference image
   - Step-by-step treatment plan
   - Recommended supplement with buy link
4. **Library** – Browse all 39 diseases in the Disease Library section

---

### STEP 9 (Optional) – Docker Deployment

```bash
# Build image
docker build -t plantguard-ai .

# Run container
docker run -p 5000:5000 plantguard-ai

# Visit: http://localhost:5000
```

---

## 🌾 Supported Plants & Diseases

| Plant       | Diseases Covered                                              |
|-------------|---------------------------------------------------------------|
| Apple       | Scab, Black Rot, Cedar Rust, Healthy                          |
| Blueberry   | Healthy                                                       |
| Cherry      | Powdery Mildew, Healthy                                       |
| Corn        | Cercospora/Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape       | Black Rot, Esca/Black Measles, Isariopsis Leaf Spot, Healthy  |
| Orange      | Huanglongbing (Citrus Greening)                               |
| Peach       | Bacterial Spot, Healthy                                       |
| Pepper Bell | Bacterial Spot, Healthy                                       |
| Potato      | Early Blight, Late Blight, Healthy                            |
| Raspberry   | Healthy                                                       |
| Soybean     | Healthy                                                       |
| Squash      | Powdery Mildew                                                |
| Strawberry  | Leaf Scorch, Healthy                                          |
| Tomato      | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🧠 Model Architecture

```
Input Image (224×224×3)
     │
Rescaling (÷255) + Augmentation
     │
MobileNetV2 Backbone (pretrained on ImageNet)
     │
GlobalAveragePooling2D
     │
Dense(256, ReLU) + Dropout(0.2)
     │
Dense(39, Softmax)
     │
Output: 39-class probabilities
```

---

## 🛠️ Tech Stack

| Component   | Technology          |
|-------------|---------------------|
| ML Model    | TensorFlow / Keras  |
| Backbone    | MobileNetV2         |
| Dataset     | PlantVillage (Kaggle)|
| Backend     | Flask (Python)      |
| Frontend    | HTML5, CSS3, Vanilla JS |
| Container   | Docker              |
| Data        | disease_info.csv, supplement_info.csv |

---

## 📌 Tips for Best Accuracy

- Use **clear, close-up** photos of affected leaves
- Ensure **good lighting** – avoid shadows
- **Single leaf** per image gives best results
- **Avoid blurry** images (model confidence will be low)

---

## ⚠️ Disclaimer

This tool is for educational and informational purposes. For critical agricultural decisions, always consult a certified agronomist or plant pathologist.
