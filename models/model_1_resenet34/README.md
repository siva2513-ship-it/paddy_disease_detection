## 2️⃣ Model 1 — ResNet34 README

Replace `models/model_1_resenet34/README.md` with:

````markdown
# Model 1 — ResNet34 CNN

ResNet34 is the **first and primary deep learning model** in this project.

It is used to classify paddy crop images into 10 disease/pest categories. More importantly, the trained ResNet34 is also used later as a **feature extractor** for the classical machine learning experiments.

The model is based on a pretrained ResNet34 network and is fine-tuned on the Paddy Doctor dataset using FastAI with a PyTorch backend.

---

## 🎯 Why ResNet34?

The first goal of the project was to establish a strong deep-learning baseline before experimenting with traditional machine learning algorithms.

ResNet34 was selected as the baseline because it provides a relatively deep convolutional architecture with pretrained ImageNet weights, allowing the model to start from useful visual representations and then adapt them to paddy disease images.

The model is trained directly on the images rather than relying on manually designed image features.

---

## 🧠 Role in the Project

ResNet34 has two important roles:

### 1. Direct Image Classifier

The network receives a paddy crop image and predicts one of the 10 classes.

```text
Paddy Leaf Image
       ↓
   ResNet34 CNN
       ↓
  Learned Features
       ↓
  Classification
       ↓
   10 Classes
````

### 2. Feature Extractor

After training, the final classification layer is removed for the feature-extraction stage.

The remaining network produces a **512-dimensional feature vector** for each image.

```text
Paddy Leaf Image
       ↓
   ResNet34
       ↓
512-D Feature Vector
       ↓
 ┌─────┬─────┬──────────────┬─────────┐
 ↓     ↓     ↓              ↓
 SVM  KNN  Random Forest  XGBoost
```

This allows the same learned visual representation to be evaluated using traditional ML algorithms.

---

# 🏗️ Architecture

The model uses:

**ResNet34 + FastAI + PyTorch**

The ResNet34 network is initialized with pretrained ImageNet weights and then fine-tuned for the 10-class paddy disease classification task.

The classifier head produces predictions for:

```text
10 classes
```

The trained model's representation immediately before the final classifier provides the **512-dimensional feature vector** used by Models 2–5.

---

# 📷 Data Preprocessing

The preprocessing pipeline follows the implementation used in the training notebook.

### Initial Resize

Images are resized to:

```text
480 px
```

using the `squish` resize method.

### Training Transform

The images are then transformed to:

```text
224 × 224
```

using FastAI's `aug_transforms` with:

```text
min_scale = 0.75
```

This provides data augmentation during CNN training.

---

# ⚙️ Training Configuration

| Setting                | Value            |
| ---------------------- | ---------------- |
| Architecture           | ResNet34         |
| Framework              | FastAI / PyTorch |
| Pretrained             | ImageNet         |
| Initial resize         | 480 px           |
| Training image size    | 224 × 224        |
| Augmentation           | `aug_transforms` |
| Minimum scale          | 0.75             |
| Epochs                 | 50               |
| Learning rate          | 0.005            |
| Precision              | FP16             |
| Train/validation split | 80/20            |
| Random seed            | 42               |
| Training metric        | Error rate       |

The dataset is split into:

```text
Training   : 8,326 images
Validation : 2,081 images
```

---

# 🚀 Training

The model is trained using FastAI's `fine_tune` procedure.

The important training configuration is:

```python
learn.fine_tune(50, 0.005)
```

This means the pretrained network is adapted to the paddy disease classification task for 50 epochs using a learning rate of 0.005.

Mixed precision (`fp16`) is used during training.

---

# 📊 Verified Performance

The ResNet34 model achieved:

## **98.03% Validation Accuracy**

The result was obtained on the project's held-out validation set containing:

```text
2,081 images
```

with:

```text
Error rate = 0.0197
Accuracy   = 98.03%
```

This is currently the **best verified result in the project**.

---

# ⚠️ About the TTA Result

The notebook contains a separately written summary mentioning a TTA accuracy of approximately 98.17%.

However, that result is **not treated as verified** in this repository because the notebook does not contain a distinct executed output that reproduces those numbers.

Therefore, the official result documented for this model is:

> **98.03% validation accuracy**

rather than the unverified TTA value.

---

# 🧪 Evaluation Dataset

The model was evaluated using the project's validation split:

```text
Total labelled images : 10,407
Training             : 8,326
Validation           : 2,081
Split                : 80/20
Random seed           : 42
```

The official Kaggle test set contains 3,469 images, but those images are unlabeled in the downloaded dataset structure.

Because ground-truth labels are unavailable, a meaningful local accuracy cannot be calculated for that test set.

Therefore:

> **This model does not currently have a verified score on the official Kaggle test set.**

---

# 🔢 Feature Extraction

One of the most important uses of this model is feature extraction.

After training, the final 10-class classification layer is removed and the remaining network is used to convert each image into a numerical representation.

Each image becomes:

```text
Image
  ↓
ResNet34
  ↓
512 numbers
```

The resulting feature matrix has:

```text
Training features   : 8326 × 512
Validation features : 2081 × 512
```

These features are saved as:

```text
X_train.npy
y_train.npy
X_valid.npy
y_valid.npy
```

The four classical ML models use these saved features as their input.

---

# 💾 Model Files

This folder contains two versions of the trained ResNet34 model.

| File                          | Description                                                                  | Approx. Size |
| ----------------------------- | ---------------------------------------------------------------------------- | -----------: |
| `resnet34_paddy_baseline.pkl` | Full FastAI Learner export containing the model and associated data pipeline |       ~84 MB |
| `resnet34_paddy_baseline.pth` | PyTorch model weights/state information                                      |      ~250 MB |

### `.pkl`

The FastAI learner can be loaded using:

```python
from fastai.learner import load_learner

learn = load_learner("resnet34_paddy_baseline.pkl")
```

### `.pth`

The `.pth` file contains the saved model weights and is intended for loading into the corresponding model architecture.

---

# 📦 Git LFS

Because the trained model files are large, they are stored using **Git Large File Storage (Git LFS)**.

After cloning the repository:

```bash
git lfs install
git lfs pull
```

Without `git lfs pull`, the large model files may appear as small pointer files instead of the actual model binaries.

---

# 📓 Training Notebook

The model was trained in:

```text
colab_notebooks/resenet34_baseline_.ipynb
```

The notebook contains the complete workflow for:

1. Dataset download
2. Dataset preparation
3. DataLoader creation
4. ResNet34 initialization
5. Transfer learning
6. Fine-tuning
7. Validation
8. Model export

---

# 🔗 Connection to Other Models

ResNet34 is the foundation for the remaining models in the repository.

```text
                    ResNet34
                       │
                       ▼
              512-D Feature Vector
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
       SVM            KNN       Random Forest
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                    XGBoost
```

Models 2–5 therefore do not process the original images directly.

They operate on the learned representation generated by this ResNet34 model.

---

# 💡 Key Observation

The ResNet34 experiment provides an important baseline for the rest of the project.

The fine-tuned CNN achieved:

**98.03% validation accuracy**

while the best-performing classical model using its extracted features achieved:

**76.93% validation accuracy with SVM.**

This difference shows that, in the current experiment, allowing the CNN itself to learn the final classification task produced a substantially stronger result than freezing its representation and training a separate classical classifier.

The classical models are still valuable because they provide a useful comparison and help us understand how much classification performance can be achieved from the learned CNN features alone.

---

# 🚧 Current Status

### ✅ Completed

* ResNet34 model training
* Transfer learning
* Fine-tuning for 50 epochs
* Validation evaluation
* Model export
* 512-dimensional feature extraction
* Feature files saved for downstream ML models
* Model files stored using Git LFS

### 🔄 Future Improvements

Possible next experiments include:

* Hyperparameter optimization
* Comparison with other CNN architectures
* Lightweight CNN architectures
* MobileNetV2 experimentation
* Real-world field-image validation
* Single-image inference
* Deployment as an API
* Mobile application integration

---

## 📌 Summary

ResNet34 currently serves as the **core CNN model and feature extractor** for the project.

Its verified validation performance is:

> **98.03% accuracy on 2,081 validation images**

The trained network is then reused to generate 512-dimensional feature vectors, which form the input for the project's classical machine learning experiments.

```

This version keeps the technically important details from the existing ResNet34 README—especially the **50 epochs, 0.005 learning rate, 8,326/2,081 split, 512-D features, and verified 98.03% result**—but explains *why* each part matters rather than just listing it. :contentReference[oaicite:0]{index=0}
```
