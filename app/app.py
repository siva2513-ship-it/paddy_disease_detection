import os
import io
import gc
import base64
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet34
from fastai.vision.learner import create_vision_model
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

RESNET_PTH = MODEL_DIR / "model_1_resenet34" / "resnet34_paddy_baseline.pth"
SVM_PATH = MODEL_DIR / "model_2_svm" / "model_2_svm.pkl"
KNN_PATH = MODEL_DIR / "model_3_knn" / "model_3_knn.pkl"
RF_PATH = MODEL_DIR / "model_4_random_forest" / "model_4_random_forest.pkl"
XGB_PATH = MODEL_DIR / "model_5_xgboost" / "model_5_xgboost.pkl"

CLASSES = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

CLASS_DISPLAY_NAMES = {
    'bacterial_leaf_blight': 'Bacterial Leaf Blight',
    'bacterial_leaf_streak': 'Bacterial Leaf Streak',
    'bacterial_panicle_blight': 'Bacterial Panicle Blight',
    'blast': 'Rice Blast',
    'brown_spot': 'Brown Spot',
    'dead_heart': 'Dead Heart (Stem Borer)',
    'downy_mildew': 'Downy Mildew',
    'hispa': 'Rice Hispa',
    'normal': 'Healthy Paddy Crop',
    'tungro': 'Rice Tungro Virus'
}

DISEASE_INFO = {
    'bacterial_leaf_blight': {
        'symptoms': 'Water-soaked lesions on leaf margins turning yellow to whiteish-grey.',
        'treatment': 'Apply copper oxychloride or streptomycin sulphate. Avoid excess nitrogen.'
    },
    'bacterial_leaf_streak': {
        'symptoms': 'Narrow, dark translucent streaks along leaf veins that turn brown.',
        'treatment': 'Spray copper-based bactericides during early leaf streak emergence.'
    },
    'bacterial_panicle_blight': {
        'symptoms': 'Panicle discoloration, aborted florets, and brown grain lesions.',
        'treatment': 'Seed dressing with copper compounds or oxolinic acid before sowing.'
    },
    'blast': {
        'symptoms': 'Spindle- or eye-shaped grey-centered spots on leaves and neck rot.',
        'treatment': 'Spray Tricyclazole 75 WP or Isoprothiolane at first onset of blast lesions.'
    },
    'brown_spot': {
        'symptoms': 'Oval brown spots with dark margins and yellow halos across leaf surfaces.',
        'treatment': 'Apply Mancozeb or Edifenphos foliar spray.'
    },
    'dead_heart': {
        'symptoms': 'Drying and death of central tiller shoots caused by Stem Borer larvae.',
        'treatment': 'Apply Cartap Hydrochloride 4G or Chlorantraniliprole granules.'
    },
    'downy_mildew': {
        'symptoms': 'Stunted seedling growth, yellow leaf flecking, and white downy fungal coating.',
        'treatment': 'Treat seeds with Metalaxyl 35 SD before planting.'
    },
    'hispa': {
        'symptoms': 'Parallel white translucent patches caused by adult beetles scraping leaf tissue.',
        'treatment': 'Spray Chlorpyrifos 20 EC or Neem seed kernel extract (NSKE 5%).'
    },
    'normal': {
        'symptoms': 'Vibrant green leaves, uniform tiller growth, and healthy plant structure.',
        'treatment': 'No treatment required. Continue standard agricultural practices.'
    },
    'tungro': {
        'symptoms': 'Stunted growth, yellow-orange leaf discoloration, and reduced tillering.',
        'treatment': 'Control green leafhopper vectors using Imidacloprid 17.8 SL.'
    }
}

MODEL_NAMES = {
    'resnet34': 'ResNet34 (CNN Baseline - 98.03% Acc)',
    'svm': 'Support Vector Machine (SVM - 76.93% Acc)',
    'xgboost': 'XGBoost (76.66% Acc)',
    'knn': 'k-Nearest Neighbors (k-NN - 72.68% Acc)',
    'rf': 'Random Forest (69.84% Acc)'
}

# Lazy loading cache dictionary
MODEL_CACHE = {}

transform_pipeline = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_resnet_and_extractor():
    if 'resnet_model' not in MODEL_CACHE:
        print("--> Loading ResNet34 PyTorch Model into memory...")
        resnet = create_vision_model(resnet34, n_out=len(CLASSES))
        state = torch.load(RESNET_PTH, map_location='cpu', weights_only=False)
        resnet.load_state_dict(state['model'])
        resnet.eval()

        extractor = torch.nn.Sequential(resnet[0], resnet[1][:5])
        extractor.eval()

        MODEL_CACHE['resnet_model'] = resnet
        MODEL_CACHE['feat_extractor'] = extractor
        gc.collect()

    return MODEL_CACHE['resnet_model'], MODEL_CACHE['feat_extractor']

def get_classical_model(model_key):
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]

    print(f"--> Lazy loading classical model: {model_key}...")
    if model_key == 'svm':
        model = joblib.load(SVM_PATH)
    elif model_key == 'knn':
        model = joblib.load(KNN_PATH)
    elif model_key == 'rf':
        model = joblib.load(RF_PATH)
    elif model_key == 'xgboost':
        model = joblib.load(XGB_PATH)
    else:
        model = None

    MODEL_CACHE[model_key] = model
    gc.collect()
    return model

@app.route('/')
def index():
    return render_template('index.html', models=MODEL_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    selected_model_key = request.form.get('model', 'resnet34')

    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        img_tensor = transform_pipeline(pil_img).unsqueeze(0)

        resnet_model, feat_extractor = get_resnet_and_extractor()

        pred_class_key = None
        confidence = "N/A"

        if selected_model_key == 'resnet34':
            with torch.no_grad():
                outputs = resnet_model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            pred_class_key = CLASSES[pred_idx]
            confidence = f"{float(probs[pred_idx]) * 100:.2f}%"
        else:
            with torch.no_grad():
                feats = feat_extractor(img_tensor).numpy()

            model = get_classical_model(selected_model_key)
            pred_idx = int(model.predict(feats)[0])
            pred_class_key = CLASSES[pred_idx]

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(feats)[0]
                confidence = f"{float(probs[pred_idx]) * 100:.2f}%"

        info = DISEASE_INFO[pred_class_key]

        return jsonify({
            'success': True,
            'image_data': f"data:image/jpeg;base64,{img_base64}",
            'model_used': MODEL_NAMES.get(selected_model_key, selected_model_key),
            'disease': CLASS_DISPLAY_NAMES[pred_class_key],
            'confidence': confidence,
            'symptoms': info['symptoms'],
            'treatment': info['treatment']
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"\n--> Paddy Disease Detection Server Running on http://127.0.0.1:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
